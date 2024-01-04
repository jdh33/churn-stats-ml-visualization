import numpy as np
import pandas as pd


""" Data cleanup and processing utilities """
def fill_nan_with_median_by_dtype(df):
    for col in df:
        data_type = df[col].dtype
        if data_type == np.int64:
            df[col] = df[col].fillna(
                df[col].median().round(0).astype(np.int64))
        elif data_type == np.float64:
            df[col] = df[col].fillna(df[col].median())

def fill_nan_with_constant_by_dtype(df):
    for col in df:
        data_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(data_type):
            df[col] = df[col].fillna(0)
        elif pd.api.types.is_string_dtype(data_type):
            df[col] = df[col].fillna('')

def assign_quartile(row, col, quartiles):
    row_value = row[col]
    if row_value <= quartiles[0.25]:
        return 0
    elif quartiles[0.25] < row_value <= quartiles[0.5]:
        return 1
    elif quartiles[0.5] < row_value <= quartiles[0.75]:
        return 2
    elif row_value > quartiles[0.75]:
        return 3
    else:
        return np.nan


""" Data exploration utilities """
cramers_v_cutoffs = np.array(
    [
        [1, .1, .3, .5],
        [2, .07, .21, .35],
        [3, .06, .17, .29],
        [4, .05, .15, .25],
        [5, .04, .13, .22]
        ])
cramers_v_columns = ['Degrees of freedom', 'Small', 'Medium', 'Large']
cramers_v_table = pd.DataFrame(
    data=cramers_v_cutoffs, columns=cramers_v_columns)
cramers_v_table = cramers_v_table.set_index(['Degrees of freedom'], drop=True)

def calculate_cramers_v(df_crosstab, chi2_statistic):
    '''
    Modefied from https://towardsdatascience.com/
    contingency-tables-chi-squared-and-cramers-v-ada4f93ec3fd
    Return the effect size of the Chi^2 test for independence
    '''
    n = df_crosstab.sum().sum()
    # The Cramer's V degrees of freedom is different than the Chi^2
    # degrees of freedom
    dof = min(df_crosstab.shape) - 1
    effect_size = np.sqrt(chi2_statistic / (n * dof))
    interpretation = 'NA'
    if effect_size >= cramers_v_table.loc[dof, 'Large']:
        interpretation = 'Large'
    elif effect_size >= cramers_v_table.loc[dof, 'Medium']:
        interpretation = 'Medium'
    elif effect_size >= cramers_v_table.loc[dof, 'Small']:
        interpretation = 'Small'
    else:
        interpretation = 'Negligible'
    return (effect_size, interpretation)

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"
