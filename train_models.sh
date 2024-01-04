#!/bin/bash
VARIABLES_TO_DROP="MonthlyCharges,TotalCharges,Tenure"
N_JOBS=1

python src/models/train_classification_models.py --training_dataset "telco-customer-churn" \
        --target_variable "Churn" --variables_to_drop $VARIABLES_TO_DROP \
        --n_jobs=$N_JOBS
