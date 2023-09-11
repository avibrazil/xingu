import argparse
import logging

from . import DataProviderFactory
from . import Coach
from . import Model
from . import ConfigManager



def prepare_logging(level=logging.INFO):
    # Switch between INFO/DEBUG while running in production/developping:

    # Configure logging for PanModel

    FORMATTER = logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s")
    HANDLER = logging.StreamHandler()
    HANDLER.setFormatter(FORMATTER)

    loggers=[
        logging.getLogger('__main__'),
        logging.getLogger('xingu'),
        logging.getLogger('sqlite')
    ]

    for logger in loggers:
        logger.addHandler(HANDLER)
        logger.setLevel(level)

    return loggers[0]



def prepare_args():
    parser = argparse.ArgumentParser(
        prog='xingu',
        description='Train and Batch Predict various Models for multiple DataProviders'
    )

    parser.add_argument('--dps', '--data-providers', dest='DATAPROVIDER_LIST', required=False,
        default=ConfigManager().get('DATAPROVIDER_LIST', default=None),
        help='Comma separated list of DataProvider IDs to process. Overwrites DATAPROVIDER_LIST env. If not set, all classes inherited from DataProvider that have IDs will be used and a Model for each will be trained.')

    parser.add_argument('--dps-folder', '--data-providers-folder', dest='DATAPROVIDER_FOLDER', required=False,
        default=ConfigManager().get('DATAPROVIDER_FOLDER', default=None),
        help='Folder name to scan for DataProviders. DataProviders are classes that hinerit from DataProvider and have an id set.')

    parser.add_argument('--pre-req-train-or-session-ids', dest='PRE_REQ_TRAIN_OR_SESSION_IDS', required=False,
        default=ConfigManager().get('PRE_REQ_TRAIN_OR_SESSION_IDS', default=None),
        help='Comma-separated list to train or train session ID to search for pre-req models. Overwrites PRE_REQ_TRAIN_OR_SESSION_IDS env.')

    parser.add_argument('--models-db', dest='XINGU_DB_URL', required=False,
        default=ConfigManager().get('XINGU_DB_URL',default='sqlite:///xingu.db?check_same_thread=False'),
        help='URL for Xingu´s control database as «mysql://user:pass@host.com/dbname?charset=utf8mb4». Overwrites XINGU_DB_URL env. If empty, uses sqlite:///xingu.db?check_same_thread=False, which is a SQLite database on current folder.')

    parser.add_argument('--table-prefix', '--prefix', dest='XINGU_DB_TABLE_PREFIX',
        default=ConfigManager().get('XINGU_DB_TABLE_PREFIX',default=None),
        help='A string to prefix every Xingu DB table name with, such as “avi_”. Overwrites XINGU_DB_TABLE_PREFIX env.')

    parser.add_argument('--database', nargs="+", action="append", dest='DATABASES',
        default=ConfigManager().get('DATABASES',default=None),
        help='Takes 2 arguments: nickname and SQLAlchemy URL of a database. Can be used multiple times to define multiple databases. Overwrites DATABASES env.')

    parser.add_argument('--hyperopt-strategy', dest='HYPEROPT_STRATEGY',
        default=ConfigManager().get('HYPEROPT_STRATEGY',default=None),
        help='Strategy for hyperparam optimization before training process. May be “last” or “self” or “dp” or a traind_id or a train_session_id or simply not set. Overwrites HYPEROPT_STRATEGY env. If None or not set, Estimator’s defaults will be used.')

    parser.add_argument('--datasource-cache-path', dest='DATASOURCE_CACHE_PATH',
        default=ConfigManager().get('DATASOURCE_CACHE_PATH',default=None),
        help='Folder to store parquets of unprocessed DataProviders’ SQL queries results. Useful to speed up consecutive and repetitive runs in development scenarios. Overwrites DATASOURCE_CACHE_PATH env.')

    parser.add_argument('--dvc-query-cache-path', dest='DVC_DATASOURCE_CACHE_PATH',
        default=ConfigManager().get('DVC_DATASOURCE_CACHE_PATH',default=None),
        help='Usually set this to the same as --datasource-cache-path. If set, causes DVC to commit DataProvider’s queries cache files. If set to a different folder, an additional (and unnecessary) copy of cache files will be created there, just for DVC. Overwrites DVC_DATASOURCE_CACHE_PATH env.')

    parser.add_argument('--trained-models-path', dest='TRAINED_MODELS_PATH',
        default=ConfigManager().get('TRAINED_MODELS_PATH',default=None),
        help='A local folder or an S3 path to dump trained models’ pickles. Example: «s3://mlops-data/sample-dvc/teste-do-avi/avm-trained-models». Overwrites TRAINED_MODELS_PATH env.')

    parser.add_argument('--dvc-trained-models-path', dest='DVC_TRAINED_MODELS_PATH',
        default=ConfigManager().get('DVC_TRAINED_MODELS_PATH',default=None),
        help='A local path to dump trained model pickles to later be added to DVC. Overwrites DVC_TRAINED_MODELS_PATH env.')

    parser.add_argument('--train', dest='TRAIN', action=argparse.BooleanOptionalAction,
        default=ConfigManager().get('TRAIN',default=True, cast=bool),
        help='Set if you want to train models. Overwrites TRAIN env.')

    parser.add_argument('--parallel-train-max-workers', dest='PARALLEL_TRAIN_MAX_WORKERS',
        default=ConfigManager().get('PARALLEL_TRAIN_MAX_WORKERS', default=0),
        help='How many PanModels to train in parallel taking care of dependencies. Overwrites PARALLEL_TRAIN_MAX_WORKERS env.')

    parser.add_argument('--parallel-hyperopt-max-workers', dest='PARALLEL_HYPEROPT_MAX_WORKERS',
        default=ConfigManager().get('PARALLEL_HYPEROPT_MAX_WORKERS', default=0),
        help='Control parallelism in hyper-parameter optimization. Overwrites PARALLEL_HYPEROPT_MAX_WORKERS env.')

    parser.add_argument('--post-process', dest='POST_PROCESS', action=argparse.BooleanOptionalAction,
        default=ConfigManager().get('POST_PROCESS',default=True, cast=bool),
        help='Set if you want to post-process trains (save model, save data sets, batch predict, metrics). Usefull when hyper-parameters optimization is the goal. Overwrites POST_PROCESS env.')

    parser.add_argument('--parallel-post-process-max-workers', dest='PARALLEL_POST_PROCESS_MAX_WORKERS',
        default=ConfigManager().get('PARALLEL_POST_PROCESS_MAX_WORKERS', default=0),
        help='How many PanModels to post-process in parallel. Overwrites PARALLEL_POST_PROCESS_MAX_WORKERS env.')

    parser.add_argument('--parallel-estimators-max-workers', dest='PARALLEL_ESTIMATORS_MAX_WORKERS',
        default=ConfigManager().get('PARALLEL_ESTIMATORS_MAX_WORKERS', default=0),
        help='How many estimators to train in parallel. Overwrites PARALLEL_ESTIMATORS_MAX_WORKERS env.')

    parser.add_argument('--batch-predict', dest='BATCH_PREDICT', action=argparse.BooleanOptionalAction,
        default=ConfigManager().get('BATCH_PREDICT',default=True, cast=bool),
        help='Set if you want to batch predict data, which is required for many metrics computations. If --no-train, use pre-trained models from TRAINED_MODELS_PATH. Overwrites BATCH_PREDICT env.')

    parser.add_argument('--dvc', dest='DVC', action=argparse.BooleanOptionalAction,
        default=ConfigManager().get('DVC',default=True, cast=bool),
        help='If --no-dvc, disables all DVC operations regardless of what other DVC-related variables have. Overwrites DVC env.')

    parser.add_argument('--commit-inventory', dest='COMMIT_INVENTORY', action=argparse.BooleanOptionalAction,
        default=ConfigManager().get('COMMIT_INVENTORY',default=False, cast=bool),
        help='The inventory.yaml file will be updated at the end of train session. This flag controls weather it should be also commited to Git’s current branch and pushed to remote repo. Overwrites COMMIT_CURRENTS env.')

    parser.add_argument('--project-home', dest='PROJECT_HOME',
        default=ConfigManager().get('PROJECT_HOME',default='.'),
        help='Local clone of xingu git repository, to collect various metadata during runtime. Overwrites PROJECT_HOME env.')


    parser.add_argument('--debug', dest='DEBUG', action=argparse.BooleanOptionalAction,
        default=ConfigManager().get('DEBUG',default=False, cast=bool),
        help='Be more verbose and output messages to console.')

    parsed = parser.parse_args()

    unset_if_none=[
        'DATAPROVIDER_LIST',            'XINGU_DB_TABLE_PREFIX',
        'QUERY_CACHE_PATH',             'DVC_QUERY_CACHE_PATH',
        'DVC_TRAINED_MODELS_PATH',      'PARALLEL_TRAIN_MAX_WORKERS',
        'HYPEROPT_STRATEGY',            'TRAINED_MODELS_PATH',
        'PRE_REQ_TRAIN_OR_SESSION_IDS',
    ]

    args={}
    for c in parsed.__dict__:
        if parsed.__dict__[c] is not None or c not in unset_if_none:
            args.update({c: parsed.__dict__[c]})

    return args




def main():
    # Read environment and command line parameters
    args=prepare_args()


    # Setup logging
    global logger
    if args['DEBUG']:
        logger=prepare_logging(logging.DEBUG)
    else:
        logger=prepare_logging()

    # Gather all DataProviders requested
    if 'DATAPROVIDER_LIST' in args:
        dpf=DataProviderFactory(
            providers_list=[x.strip() for x in args['DATAPROVIDER_LIST'].split(',')],
            providers_folder=args['DATAPROVIDER_FOLDER']
        )
    else:
        dpf=DataProviderFactory(providers_folder=args['DATAPROVIDER_FOLDER'])

    # Prepare list of databases to be parsed latter by Coach.get_db_connection()
    if isinstance(args['DATABASES'],list):
        args['DATABASES'] = '|'.join(args['DATABASES'])

    # Control DVC
    if args['DVC'] == False:
        dvc_killer=['DVC_QUERY_CACHE_PATH', 'DVC_TRAINED_MODELS_PATH']
        for dvc in dvc_killer:
            if dvc in args: del args[dvc]


    # Now set the environment with everything useful that we got
    ConfigManager().set(args)


    # Setup a coach
    coach=Coach(dp_factory = dpf)


    if args['TRAIN']:
        coach.team_train()
    elif args['BATCH_PREDICT']:
        coach.team_batch_predict()
    else:
        logger.info('Please use both or one of --batch-preditc and --train')





if __name__ == "__main__":
    main()
