#!/bin/sh

# Train or deploy Robson multiple times using different code branches.
# After training, find train_ids associated with each branch querying the
# database as:
#
#     SELECT dataprovider_id,train_session_id,train_id,git_branch
#     FROM training
#     WHERE git_branch like "%feat/my-desired-git-branch"
#
# Avi Alkalay
# 2022-08-19
#

# List of branches you want to train (space separated)
branches="feat/release-setembro-2022"


# List of DPs to train
export DATAPROVIDER_LIST="vitrine_bh, vitrine_rj, vitrine_sp"
# export DATAPROVIDER_LIST="vitrine_bh"
# export DATAPROVIDER_LIST="anuncios_scs, anuncios_gru"

# Parameters to optimize and train Robson database
export DEBUG=true
export PROJECT_HOME=.
export QUERY_CACHE_PATH=data
# export TRAINED_MODELS_PATH=models
export TRAINED_MODELS_PATH="s3://{%AWS_PARAM:robson-avm-staging-bucket%}/trained-models"
# export ROBSON_DB_URL="sqlite:///robson.db?check_same_thread=False"
export ROBSON_DB_URL="postgresql+psycopg2://{%AWS_PARAM:robson-avm-staging-user%}:{%AWS_SECRET:robson-avm-staging-rds-secret%}@{%AWS_PARAM:robson-avm-staging-url%}/{%AWS_PARAM:robson-avm-staging-database-name%}"
export DATALAKE_DATABRICKS_URL="databricks+connector://token:dapi170fe70c366410b94bc76d2082ca01a3@dbc-da926df9-ab65.cloud.databricks.com/default?http_path=/sql/1.0/endpoints/b49aee71843b4d3e"
export PRE_REQ_TRAIN_OR_SESSION_IDS="accepting-turret"

# If HYPEROPT_STRATEGY=self (hyperoptimize), avoid multiple parallel trainings
# by making PARALLEL_TRAIN_MAX_WORKERS=1

# What to do and how
## Train part
export TRAIN=true
export HYPEROPT_STRATEGY="self"
export PARALLEL_TRAIN_MAX_WORKERS=1      # How many Robsons to train in parallel
export PARALLEL_HYPEROPT_MAX_WORKERS=0   # How many CPUs to let Ray use (0=all)

### Ray configuration
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_DISABLE_IMPORT_WARNING=1

### Allow Ray to run inside a thread
export TUNE_DISABLE_SIGINT_HANDLER=1




## Post process and batch predict part
export POST_PROCESS=true       # Save PKLs and batch predict
export BATCH_PREDICT=true
export PARALLEL_ESTIMATORS_MAX_WORKERS=0
export PARALLEL_POST_PROCESS_MAX_WORKERS=0

export COMMIT_CURRENTS=true


train() {
    # Train whatever the environment tells me to train
    
    python -m robson
}

deploy() {
    # Deploy between environments using currents.yaml
    # Probably needs tweaks to meet your needs.
    
    python -m robson.deploy \
        --source-robson-db $ROBSON_DB_URL \
        --source-trained-models-path $TRAINED_MODELS_PATH \
        --target-robson-db "postgresql+psycopg2://{%AWS_PARAM:robson-avm-staging-user%}:{%AWS_SECRET:robson-avm-staging-rds-secret%}@{%AWS_PARAM:robson-avm-staging-url%}/{%AWS_PARAM:robson-avm-staging-database-name%}" \
        --target-trained-models-path "s3://{%AWS_PARAM:robson-avm-staging-bucket%}/trained-models"
}

# Run Robson for each branch

current_branch=`git branch --show current`

for b in $branches; do
    echo "Switching to branch $b..."
    git switch "$b"
    
    # Action to execute, implemented as shell functions above
    train
    # deploy

    # Collect and report some results
    result=$?
    sleep 5
    printf "\n\n\n\n"
    echo "Exit status for branch $b: $result"
    printf "\n\n\n\n"
done 2>&1 | tee robson-multibranch.log

git switch "$current_branch"

