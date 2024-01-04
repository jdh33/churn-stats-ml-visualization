#!/bin/bash

N_JOBS=1

python scripts/train_models.py --training_dataset "telco-customer-churn/processed/" \
        --target_variable "Churn" --train_with_shuffled_data \
        --timestamp --n_jobs=$N_JOBS
