import sys
import os
import datetime
import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, minmax_scale
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.inspection import permutation_importance
from joblib import dump

sys.path.append(os.path.join('src', 'utilities'))
import utilities as utils

def main():
    args = utils.get_classification_model_training_args()
    training_dataset = args.training_dataset
    target_variable = args.target_variable
    ordinal_variables = args.ordinal_variables.split(',')
    variables_to_drop = args.variables_to_drop.split(',')
    # TODO: add support for other models
    # selected_models = args.selected_models.split(',')
    train_with_shuffled_data = args.train_with_shuffled_data
    output_suffix = args.output_suffix
    timestamp = args.timestamp
    n_jobs = args.n_jobs

    if not ordinal_variables[0].strip():
        ordinal_variables = []
    if not variables_to_drop[0].strip():
        variables_to_drop = []
    output_dir = training_dataset
    if output_suffix.strip():
        output_dir = f'{output_dir}_{output_suffix}'
    if timestamp:
        run_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        output_dir = f'{output_dir}_{run_datetime}'
    
    # This will be the directory of the shell script
    project_dir = os.path.abspath(os.getcwd())
    ml_config_path = os.path.join(project_dir, 'configs', 'ml_config.json')
    with open (ml_config_path) as json_config:
        ml_config_options = json.load(json_config)
    ml_config = ml_config_options['grid_search_cv']

    TRAIN_TEST_SPLIT_RANDOM_STATE = 0
    CV_RANDOM_STATE = 21
    MODEL_RANDOM_STATE = 42
    N_SPLITS = 5
    FEATURE_PERMUTATION_TEST_N = 100
    VERBOSE = 1

    input_dir = os.path.join(
        project_dir, 'data', training_dataset, 'processed')
    results_dir = os.path.join(project_dir, 'models', output_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    else:
        sys.exit('Model directory already exists.')

    # Create X, X_cov, and y matrices
    df = pd.read_csv(os.path.join(input_dir, 'df.csv'), index_col=0)
    variables_to_drop.append(target_variable)
    y = df[target_variable].replace({'No': 0, 'Yes': 1}).astype(np.int64)
    X = df.drop(variables_to_drop, axis=1)
    training_variables = set(X.columns)
    nominal_variables = list(training_variables.difference(ordinal_variables))
    # Use when creating table of feature importances
    # Make sure there will be no duplicate column names if 
    # excluding a prefix
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE)
    Xy_train_test_matrices = {'X': (X_train, X_test, y_train, y_test)}
    if train_with_shuffled_data:
        X_train_shuffled = shuffle(X_train).reset_index(drop=True)
        X_test_shuffled = shuffle(X_test).reset_index(drop=True)
        Xy_train_test_matrices['X_shuffled'] = (
            X_train_shuffled, X_test_shuffled,
            y_train, y_test,
            )
    
    # Setup pipelines, hyperparameters, etc.
    average_precision_scorer = make_scorer(average_precision_score)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    encoders = make_column_transformer(
        ('passthrough', training_variables))
    if nominal_variables and ordinal_variables:
        encoders = make_column_transformer(
            (OneHotEncoder(), nominal_variables),
            (OrdinalEncoder(), ordinal_variables),
            remainder='drop')
    elif nominal_variables and not ordinal_variables:
        encoders = make_column_transformer(
            (OneHotEncoder(), nominal_variables),
            remainder='drop')
    elif not nominal_variables and ordinal_variables:
        encoders = make_column_transformer(
            (OrdinalEncoder(), ordinal_variables),
            remainder='drop')
    # Random forest
    rf_classifier = RandomForestClassifier(
        class_weight='balanced', random_state=MODEL_RANDOM_STATE)
    rf_classifier_pipeline = Pipeline(
        steps=[
            ('encoder', encoders),
            ('classifier', rf_classifier)
            ])
    rf_classifier_param_grid = {
        **ml_config['rfc']
        }
    rf_classifier_cv = GridSearchCV(
        estimator=rf_classifier_pipeline, param_grid=rf_classifier_param_grid,
        cv=cv, refit=True, scoring=average_precision_scorer, n_jobs=n_jobs,
        verbose=VERBOSE)
    models_to_train = {
        'rfc': ('Random forest classification', rf_classifier_cv)
        }
    
    all_cv_results = pd.DataFrame()
    test_result_columns = [
        'model_key', 'model_name', 'training_x',
        'ap_score', 'shuffled_ap_score', 'best_hyperparameters'
        ]
    all_test_results = pd.DataFrame(columns=test_result_columns)
    for model_cv_key in models_to_train:
        model_cv_name, model_cv = models_to_train[model_cv_key]
        for Xy_key, Xy_matrices in Xy_train_test_matrices.items():
            model_results_dir = os.path.join(
                results_dir, f'{model_cv_key}_{Xy_key}')
            if not os.path.exists(model_results_dir):
                os.makedirs(model_results_dir)
            X_train_i, X_test_i, y_train_i, y_test_i = Xy_matrices
            # Train model
            print(f'*****\n{model_cv_name} training on {Xy_key}')
            model_cv.fit(X=X_train_i, y=y_train_i)
            # Results of hyperparameter search
            cv_results = pd.DataFrame(model_cv.cv_results_)
            cv_results.insert(0, 'model_key', model_cv_key)
            cv_results.insert(1, 'model_name', model_cv_name)
            cv_results.insert(2, 'training_x', Xy_key)
            all_cv_results = pd.concat(
                [
                    all_cv_results,
                    cv_results.sort_values(by=['rank_test_score']).iloc[0:10]
                    ],
                sort=True, ignore_index=True)
            # Test on unseen data
            y_test_predictions = model_cv.predict_proba(X_test_i)[:, 1]
            y_test_predictions = pd.DataFrame(
                {'true': y_test_i, 'predicted_prob': y_test_predictions})
            test_score = average_precision_score(
                y_test_predictions['true'],
                y_test_predictions['predicted_prob'])
            shuffled_test_score = np.nan
            # Do a couple more tests with the 'standard' X matrix
            if Xy_key == 'X':
                shuffled_y_test_predictions = model_cv.predict_proba(
                    shuffle(X_test_i).reset_index(drop=True))[:, 1]
                shuffled_y_test_predictions = pd.DataFrame(
                    data={
                        'true': y_test_i,
                        'predicted_prob': shuffled_y_test_predictions
                        })
                shuffled_test_score = average_precision_score(
                    shuffled_y_test_predictions['true'],
                    shuffled_y_test_predictions['predicted_prob'])
                permutation_results = permutation_importance(
                    model_cv, X_test_i, y_test_i,
                    n_repeats=FEATURE_PERMUTATION_TEST_N,
                    random_state=MODEL_RANDOM_STATE, n_jobs=n_jobs)
                permutation_importances = {}
                for k, v in permutation_results.items():
                    # importances is an array of arrays
                    # Each subarray contains importance values for each
                    # perumattion test; we don't need all those values
                    if k != 'importances':
                        v = list(v)
                        permutation_importances[k] = v
                permutation_importances = pd.DataFrame(
                    data=permutation_importances)
                permutation_importances['normalized_importance'] = (
                    minmax_scale(permutation_importances['importances_mean']))
                permutation_importances['rank_importance'] = (
                    permutation_importances['normalized_importance'].rank(
                        ascending=False))
                permutation_importances.insert(
                    0, 'model_key', model_cv_key)
                permutation_importances.insert(
                    1, 'model_name', model_cv_name)
                permutation_importances.insert(
                    2, 'training_x', Xy_key)
                permutation_importances.insert(
                    3, 'feature', model_cv.feature_names_in_.tolist())
                permutation_importances = permutation_importances.sort_values(
                    by=['rank_importance'])
                permutation_importances.to_csv(
                    os.path.join(model_results_dir,
                                 'permutation_importances.csv'))
            test_results = [
                model_cv_key, model_cv_name, Xy_key,
                test_score, shuffled_test_score, str(model_cv.best_params_)
                ]
            test_results = pd.DataFrame(
                index=[0], data=dict(zip(test_result_columns, test_results)))
            all_test_results = pd.concat(
                [all_test_results, test_results], ignore_index=True)
            
            dump(
                model_results_dir,
                os.path.join(model_results_dir, 'model.joblib'))
            y_test_predictions.to_csv(
                os.path.join(model_results_dir, 'y_test_predictions.csv'))

    all_cv_results.to_csv(os.path.join(results_dir, 'cv_results.csv'))
    all_test_results.to_csv(os.path.join(results_dir, 'test_results.csv'))

if __name__ == '__main__':
    main()