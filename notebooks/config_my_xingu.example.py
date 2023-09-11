# import pygit2
import pathlib

# project_home should resolve to something like /home/robson/robson_avm or the root of the git project in your disk
# project_home=pathlib.PurePath(pygit2.discover_repository('.')).parents[0]
project_home=pathlib.PurePath('..') #.parents[0]

bundles = dict(
    env = dict(
        alpha_explorer=dict(
            ### DataProviders to process. Do all if empty.
            # DATAPROVIDER_LIST=              "anuncios_bh,anuncios_scs",

            ### Databases
            XINGU_DB_URL=  "sqlite:///xingu.db?check_same_thread=False",
            DATABASES=     "myathena|awsathena+rest://athena.us-east-1.amazonaws.com:443/my_db?work_group=some_workgroup&compression=snappy",

            # Athena URLs with explicit user and password work too, if you don't have ~/.aws/credentials
            # DATALAKE_ATHENA_URL="awsathena+rest://AKIAU......YZWSWWZ:YE0h......wwCzwa@athena.us-east-1.amazonaws.com:443/robson_valuation?work_group=mlops",

            ### Comma-separated list of train_ids or train_session_ids to search and pre-load pre-req estimators
            # PRE_REQ_TRAIN_OR_SESSION_IDS=   "model-one,model-two"

            ### Where to save trained models
            TRAINED_MODELS_PATH=            str(project_home / 'models'),

            ### Where to save query caches
            DATASOURCES_CACHE_PATH=               str(project_home / 'data'),

            ### Plots and graphics control
            PLOTS_PATH=                     str(project_home / 'plots'),
            PLOTS_FORMAT=                   'svg,png',

            ### Root of git project
            PROJECT_HOME=                   str(project_home),

            DEBUG=                          'true',
        ),



        beta_explorer=dict(
            ### Databases
            XINGU_DB_URL=                  "postgresql+psycopg2://{%AWS_PARAM:staging-user%}:{%AWS_SECRET:staging-rds-secret%}@{%AWS_PARAM:staging-url%}/{%AWS_PARAM:staging-database-name%}",
            DATABASES=     "myathena|awsathena+rest://athena.us-east-1.amazonaws.com:443/my_db?work_group=some_workgroup&compression=snappy",

            ### Comma-separated list of train_ids or train_session_ids to search and pre-load pre-req estimators
            # PRE_REQ_TRAIN_OR_SESSION_IDS=   "model-one,model-two"

            ### Where to save trained models
            TRAINED_MODELS_PATH=            's3://{%AWS_PARAM:staging-bucket%}/trained-models',
            DVC_TRAINED_MODELS_PATH=        str(project_home / 'models'),

            ### Where to save query caches
            DATASOURCES_CACHE_PATH=         str(project_home / 'data'),
            DVC_QUERY_CACHE_PATH=           str(project_home / 'data'),

            ### Plots and graphics control
            PLOTS_PATH=                     str(project_home / 'plots'),
            PLOTS_FORMAT=                   'svg,png',

            ### Root of git project
            PROJECT_HOME=                   str(project_home),

            COMMIT_CURRENTS=                'true',
            DEBUG=                          'true',
        ),



        staging=dict(
            ### Databases
            XINGU_DB_URL=                  "postgresql+psycopg2://{%AWS_PARAM:staging-user%}:{%AWS_SECRET:staging-rds-secret%}@{%AWS_PARAM:staging-url%}/{%AWS_PARAM:staging-database-name%}",
            DATALAKE_ATHENA_URL=            "awsathena+rest://athena.us-east-1.amazonaws.com:443/my_db?work_group=some_workgroup&compression=snappy",
            DATALAKE_DATABRICKS_URL=        "databricks+connector://token:dapi170...1a3@dbc-da9-ab6.cloud.databricks.com/default?http_path=/sql/1.0/endpoints/b4...d3e",

            ### Where to save trained models
            TRAINED_MODELS_PATH=            's3://{%AWS_PARAM:staging-bucket%}/trained-models',
            DVC_TRAINED_MODELS_PATH=        str(project_home / 'models'),

            ### Where to save query caches
            # QUERY_CACHE_PATH=               str(project_home / 'data'),
            DVC_QUERY_CACHE_PATH=           str(project_home / 'data'),

            ### Plots and graphics control
            PLOTS_PATH=                     str(project_home / 'plots'),
            PLOTS_FORMAT=                   'svg,png',

            ### Root of git project
            PROJECT_HOME=                   str(project_home),

            COMMIT_CURRENTS=                'true',
            DEBUG=                          'true',
        ),



        production=dict(
            ### Databases
            XINGU_DB_URL=                  "postgresql+psycopg2://{%AWS_PARAM:production-user%}:{%AWS_SECRET:production-rds-secret%}@{%AWS_PARAM:production-url%}/{%AWS_PARAM:production-database-name%}",
            DATALAKE_ATHENA_URL=            "awsathena+rest://athena.us-east-1.amazonaws.com:443/my_db?work_group=some_workgroup&compression=snappy",
            DATALAKE_DATABRICKS_URL=        "databricks+connector://token:dapi170...1a3@dbc-da9-ab6.cloud.databricks.com/default?http_path=/sql/1.0/endpoints/b4...d3e",

            ### Where to save trained models
            TRAINED_MODELS_PATH=            's3://{%AWS_PARAM:production-bucket%}/trained-models',
            DVC_TRAINED_MODELS_PATH=        str(project_home / 'models'),

            ### Where to save query caches
            # QUERY_CACHE_PATH=               str(project_home / 'data'),
            DVC_QUERY_CACHE_PATH=           str(project_home / 'data'),

            ### Plots and graphics control
            PLOTS_PATH=                     str(project_home / 'plots'),
            PLOTS_FORMAT=                   'svg,png',

            ### Root of git project
            PROJECT_HOME=                   str(project_home),

            COMMIT_CURRENTS=                'true',
            DEBUG=                          'false',
        )
    ),







    parallel = dict(
        train_and_predict=dict(

            # Everyday train and batch predict

            TRAIN=                                  'true',
            POST_PROCESS=                           'true',
            BATCH_PREDICT=                          'true',
            HYPEROPT_STRATEGY=                      'last',

            # Maximum parallelism.
            PARALLEL_TRAIN_MAX_WORKERS=             '0',
            PARALLEL_POST_PROCESS_MAX_WORKERS=      '0',
            PARALLEL_ESTIMATORS_MAX_WORKERS=        '0'
        ),

        predict_only=dict(

            # Everyday Batch Predict only with pre-trained models

            TRAIN=                                  'false',
            POST_PROCESS=                           'false',
            BATCH_PREDICT=                          'true',

            # Maximum parallelism.
            PARALLEL_POST_PROCESS_MAX_WORKERS=      '0'
        ),

        hyper_optimize_only=dict(

            # Hyper-parameters optimization only

            TRAIN=                                  'true',
            POST_PROCESS=                           'false',
            BATCH_PREDICT=                          'false',
            HYPEROPT_STRATEGY=                      'self',

            # One at a time
            PARALLEL_TRAIN_MAX_WORKERS=             '1',

            # Maximum parallelism.
            PARALLEL_HYPEROPT_MAX_WORKERS=          '0',
            PARALLEL_ESTIMATORS_MAX_WORKERS=        '0'
        ),

        do_all=dict(

            # Do all -- avoid maximum parallelism

            TRAIN=                                  'true',
            POST_PROCESS=                           'true',
            BATCH_PREDICT=                          'true',
            HYPEROPT_STRATEGY=                      'self',


            # One at a time
            PARALLEL_TRAIN_MAX_WORKERS=             '3',

            # Do not exagerate
            PARALLEL_HYPEROPT_MAX_WORKERS=          '6',
            PARALLEL_POST_PROCESS_MAX_WORKERS=      '3',
            PARALLEL_ESTIMATORS_MAX_WORKERS=        '0'
        ),

        do_all_sequential=dict(

            # Do all sequentially (no parallelism)

            TRAIN=                                  'true',
            POST_PROCESS=                           'true',
            BATCH_PREDICT=                          'true',
            HYPEROPT_STRATEGY=                      'self',


            # One at a time
            PARALLEL_TRAIN_MAX_WORKERS=             '1',

            # Do not exagerate
            PARALLEL_HYPEROPT_MAX_WORKERS=          '1',
            PARALLEL_POST_PROCESS_MAX_WORKERS=      '1',
            PARALLEL_ESTIMATORS_MAX_WORKERS=        '1'
        )
    )
)

