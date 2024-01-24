import argparse
import json
import logging

from . import DataProvider
from . import DPSimple
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

    parser.add_argument(
        '--dps', '--data-providers',
        dest='DATAPROVIDER_LIST', required=False,
        default=ConfigManager().get('DATAPROVIDER_LIST', default=None),
        help=(
            'Comma separated list of DataProvider IDs to process. Overwrites '
            'DATAPROVIDER_LIST env. If not set, all classes inherited from '
            'DataProvider that have IDs will be used and a Model for each will '
            'be trained.'
        )
    )

    parser.add_argument(
        '--dps-folder', '--data-providers-folder',
        dest='DATAPROVIDER_FOLDER', required=False,
        default=ConfigManager().get('DATAPROVIDER_FOLDER', default=None),
        help=(
            'Folder name to scan for DataProviders. DataProviders are classes '
            'that hinerit from DataProvider and have an id set.'
        )
    )

    parser.add_argument(
        '--pre-req-train-or-session-ids',
        dest='PRE_REQ_TRAIN_OR_SESSION_IDS', required=False,
        default=ConfigManager().get('PRE_REQ_TRAIN_OR_SESSION_IDS', default=None),
        help=(
            'Comma-separated list to train or train session ID to search for '
            'pre-req models. Overwrites PRE_REQ_TRAIN_OR_SESSION_IDS env.'
        )
    )

    parser.add_argument(
        '--models-db',
        dest='XINGU_DB_URL', required=False,
        default=ConfigManager().get('XINGU_DB_URL',default='sqlite:///xingu.db?check_same_thread=False'),
        help=(
            'URL for Xingu´s control database as '
            '«mysql://user:pass@host.com/dbname?charset=utf8mb4». Overwrites '
            'XINGU_DB_URL env. If empty, uses '
            'sqlite:///xingu.db?check_same_thread=False, which is a SQLite '
            'database on current folder.'
        )
    )

    parser.add_argument(
        '--table-prefix', '--prefix',
        dest='XINGU_DB_TABLE_PREFIX',
        default=ConfigManager().get('XINGU_DB_TABLE_PREFIX',default=None),
        help=(
            'A string to prefix every Xingu DB table name with, such as '
            '“xingu_”. Overwrites XINGU_DB_TABLE_PREFIX env.'
        )
    )

    parser.add_argument(
        '--database', nargs="+",
        action="append", dest='DATABASES',
        default=ConfigManager().get('DATABASES',default=None),
        help=(
            'Takes 2 arguments: nickname and SQLAlchemy URL of a database. '
            'Can be used multiple times to define multiple databases. '
            'Overwrites DATABASES env.'
        )
    )

    parser.add_argument(
        '--hyperopt-strategy',
        dest='HYPEROPT_STRATEGY',
        default=ConfigManager().get('HYPEROPT_STRATEGY',default=None),
        help=(
            'Strategy for hyperparam optimization before training process. '
            'May be “last” or “self” or “dp” or a traind_id or a '
            'train_session_id or simply not set. Overwrites HYPEROPT_STRATEGY '
            'env. If None or not set, Estimator’s defaults will be used.'
        )
    )

    parser.add_argument(
        '--datasource-cache-path',
        dest='DATASOURCE_CACHE_PATH',
        default=ConfigManager().get('DATASOURCE_CACHE_PATH',default=None),
        help=(
            'Folder to store parquets of unprocessed DataProviders’ SQL queries '
            'results. Useful to speed up consecutive and repetitive runs in '
            'development scenarios. Overwrites DATASOURCE_CACHE_PATH env.'
        )
    )

    parser.add_argument(
        '--dvc-query-cache-path',
        dest='DVC_DATASOURCE_CACHE_PATH',
        default=ConfigManager().get('DVC_DATASOURCE_CACHE_PATH',default=None),
        help=(
            'Usually set this to the same as --datasource-cache-path. If set, '
            'causes DVC to commit DataProvider’s queries cache files. If set to '
            'a different folder, an additional (and unnecessary) copy of cache '
            'files will be created there, just for DVC. Overwrites '
            'DVC_DATASOURCE_CACHE_PATH env.'
        )
    )

    parser.add_argument(
        '--trained-models-path',
        dest='TRAINED_MODELS_PATH',
        default=ConfigManager().get('TRAINED_MODELS_PATH',default=None),
        help=(
            'A local folder or an S3 path to dump trained models’ pickles. '
            'Example: «s3://mlops-data/sample-dvc/teste-do-avi/avm-trained-models». '
            'Overwrites TRAINED_MODELS_PATH env.'
        )
    )

    parser.add_argument(
        '--plots-path',
        dest='PLOTS_PATH',
        default=ConfigManager().get('PLOTS_PATH',default=None),
        help=(
            'A local folder or an S3 path to save plots. Overwrites PLOTS_PATH '
            'env.'
        )
    )

    parser.add_argument(
        '--plots-format',
        dest='PLOTS_FORMAT',
        default=ConfigManager().get('PLOTS_FORMAT',default='png'),
        help=(
            'Comma separated list of image formats. Example: "svg,png". '
            'Defaults to PNG only. Overwrites PLOTS_FORMAT env.'
        )
    )

    parser.add_argument(
        '--dvc-trained-models-path',
        dest='DVC_TRAINED_MODELS_PATH',
        default=ConfigManager().get('DVC_TRAINED_MODELS_PATH',default=None),
        help=(
            'A local path to dump trained model pickles to later be added to '
            'DVC. Overwrites DVC_TRAINED_MODELS_PATH env.'
        )
    )

    parser.add_argument(
        '--train',
        dest='TRAIN', action=argparse.BooleanOptionalAction,
        default=ConfigManager().get('TRAIN',default=True, cast=bool),
        help='Set if you want to train models. Overwrites TRAIN env.'
    )

    parser.add_argument(
        '--train-session-id',
        dest='TRAIN_SESSION_ID',
        default=ConfigManager().get('TRAIN_SESSION_ID',default=None),
        help=(
            'Gives this training session an ID, otherwise randomname module '
            'will be used. Overwrites TRAIN_SESSION_ID env.'
        )
    )

    parser.add_argument(
        '--hostname',
        dest='HOSTNAME',
        default=ConfigManager().get('HOSTNAME',default=None),
        help=(
            'Sometimes the detected hostname is useless for reporting purposes. '
            'This is your chance to define something more useful. '
            'Overwrites HOSTNAME env.'
        )
    )

    parser.add_argument(
        '--username',
        dest='USERNAME',
        default=ConfigManager().get('USERNAME',default=None),
        help=(
            'Sometimes the detected user name is useless for reporting purposes. '
            'This is your chance to define something more useful. '
            'Overwrites USERNAME env.'
        )
    )

    parser.add_argument(
        '--parallel-train-max-workers',
        dest='PARALLEL_TRAIN_MAX_WORKERS',
        default=ConfigManager().get('PARALLEL_TRAIN_MAX_WORKERS', default=0),
        help=(
            'How many Models to train in parallel taking care of dependencies. '
            'Overwrites PARALLEL_TRAIN_MAX_WORKERS env.'
        )
    )

    parser.add_argument(
        '--parallel-hyperopt-max-workers',
        dest='PARALLEL_HYPEROPT_MAX_WORKERS',
        default=ConfigManager().get('PARALLEL_HYPEROPT_MAX_WORKERS', default=0),
        help=(
            'Control parallelism in hyper-parameter optimization. Overwrites '
            'PARALLEL_HYPEROPT_MAX_WORKERS env.'
        )
    )

    parser.add_argument(
        '--post-process',
        dest='POST_PROCESS', action=argparse.BooleanOptionalAction,
        default=ConfigManager().get('POST_PROCESS',default=True, cast=bool),
        help=(
            'Set if you want to post-process trains (save model, save data '
            'sets, batch predict, metrics). Usefull when hyper-parameters '
            'optimization is the goal. Overwrites POST_PROCESS env.'
        )
    )

    parser.add_argument(
        '--save-sets',
        dest='SAVE_SETS', action=argparse.BooleanOptionalAction,
        default=ConfigManager().get('SAVE_SETS',default=True, cast=bool),
        help=(
            'Wether to write training ‘sets’ to Xingu database table sets or not. '
            'Overwrites SAVE_SETS env.'
        )
    )

    parser.add_argument(
        '--parallel-post-process-max-workers',
        dest='PARALLEL_POST_PROCESS_MAX_WORKERS',
        default=ConfigManager().get('PARALLEL_POST_PROCESS_MAX_WORKERS', default=0),
        help=(
            'How many PanModels to post-process in parallel. Overwrites'
            'PARALLEL_POST_PROCESS_MAX_WORKERS env.'
        )
    )

    parser.add_argument(
        '--parallel-estimators-max-workers',
        dest='PARALLEL_ESTIMATORS_MAX_WORKERS',
        default=ConfigManager().get('PARALLEL_ESTIMATORS_MAX_WORKERS', default=0),
        help=(
            'How many estimators to train in parallel. Overwrites '
            'PARALLEL_ESTIMATORS_MAX_WORKERS env.'
        )
    )

    parser.add_argument(
        '--batch-predict',
        dest='BATCH_PREDICT', action=argparse.BooleanOptionalAction,
        default=ConfigManager().get('BATCH_PREDICT',default=True, cast=bool),
        help=(
            'Set if you want to batch predict data, which is required for many '
            'metrics computations. If --no-train, use pre-trained models from '
            'TRAINED_MODELS_PATH. Overwrites BATCH_PREDICT env.'
        )
    )

    parser.add_argument(
        '--batch-predict-save-estimations',
        dest='BATCH_PREDICT_SAVE_ESTIMATIONS',
        action=argparse.BooleanOptionalAction,
        default=ConfigManager().get('BATCH_PREDICT_SAVE_ESTIMATIONS',default=False, cast=bool),
        help=(
            'Set if you want to save to Xingu DB the batch predict '
            'estimations. Default is not to save. Batch predict metrics will '
            'always be saved. Overwrites BATCH_PREDICT_SAVE_ESTIMATIONS env.'
        )
    )

    parser.add_argument(
        '--dvc',
        dest='DVC', action=argparse.BooleanOptionalAction,
        default=ConfigManager().get('DVC',default=True, cast=bool),
        help=(
            'If --no-dvc, disables all DVC operations regardless of what other '
            'DVC-related variables have. Overwrites DVC env.'
        )
    )

    parser.add_argument(
        '--commit-inventory',
        dest='COMMIT_INVENTORY', action=argparse.BooleanOptionalAction,
        default=ConfigManager().get('COMMIT_INVENTORY',default=False, cast=bool),
        help=(
            'The inventory.yaml file will be updated at the end of train '
            'session. This flag controls weather it should be also commited to '
            'Git’s current branch and pushed to remote repo. Overwrites '
            'COMMIT_CURRENTS env.'
        )
    )

    parser.add_argument(
        '--project-home', dest='PROJECT_HOME',
        default=ConfigManager().get('PROJECT_HOME',default='.'),
        help=(
            'Local clone of xingu git repository, to collect various metadata '
            'during runtime. Overwrites PROJECT_HOME env.'
        )
    )



    # Simple DataProvider arguments (entire DataProvider constructed from
    # command line arguments, no coding)
    parser.add_argument(
        '--simpledp-id', dest='SIMPLEDP_ID',
        default=ConfigManager().get('SIMPLEDP_ID',default=None),
        help=('The ID of the Simple DataProvider. Overwrites SIMPLEDP_ID env.')
    )

    parser.add_argument(
        '--simpledp-train-datasource', nargs="+",
        action="append", dest='TRAIN_DATASOURCE',
        default=ConfigManager().get('TRAIN_DATASOURCE',default=None),
        help=(
            'Use multiple times. Pass URLs (http, S3, file etc) of CSVs, '
            'parquets or JSON files to define a training datasoure. Overwrites '
            'TRAIN_DATASOURCE env.'
        )
    )

    parser.add_argument(
        '--simpledp-estimator-features', dest='ESTIMATOR_FEATURES',
        default=ConfigManager().get('ESTIMATOR_FEATURES',default=None),
        help=(
            'Comma-separated list of column names for the estimator. '
            'Overwrites ESTIMATOR_FEATURES env.'
        )
    )

    parser.add_argument(
        '--simpledp-target-feature', dest='TARGET_FEATURE',
        default=ConfigManager().get('TARGET_FEATURE',default=None),
        help=(
            'Name of target column in the train datasource. Overwrites '
            'TARGET_FEATURE env.'
        )
    )

    parser.add_argument(
        '--simpledp-proba-class-index', dest='PROBA_CLASS_INDEX', type=int,
        default=ConfigManager().get('PROBA_CLASS_INDEX',default=0,cast=int),
        help=(
            'For classifiers, the index of the desired result of '
            'predict_proba(). Defaults to 0. Overwrites PROBA_CLASS_INDEX env.'
        )
    )

    parser.add_argument(
        '--simpledp-base-class', dest='BASE_CLASS',
        default=ConfigManager().get('BASE_CLASS',default=None),
        help=(
            'Python full name of a xingu.DataProvider-derived class to be used '
            'as a base class for the trained DataProvider. Overwrites '
            'BASE_CLASS env.'
        )
    )

    parser.add_argument(
        '--simpledp-estimator-class', dest='ESTIMATOR_CLASS',
        default=ConfigManager().get('ESTIMATOR_CLASS',default=None),
        help=(
            'Python full name of a xingu.Estimator-derived class, such as '
            '“xingu.estimators.xgboost_optuna.XinguXGBoostClassifier”. '
            'Overwrites ESTIMATOR_CLASS env.'
        )
    )

    parser.add_argument(
        '--simpledp-estimator-class-params', dest='ESTIMATOR_CLASS_PARAMS',
        default=ConfigManager().get('ESTIMATOR_CLASS_PARAMS',default=None),
        help=(
            'JSON text used as parameters to your xingu.Estimator-derived '
            'class. Overwrites ESTIMATOR_CLASS_PARAMS env.'
        )
    )

    parser.add_argument(
        '--simpledp-estimator-params', dest='ESTIMATOR_PARAMS',
        default=ConfigManager().get('ESTIMATOR_PARAMS',default=None),
        help=(
            'JSON text used as parameters for the underlying algorithm. '
            'Overwrites ESTIMATOR_PARAMS env.'
        )
    )

    parser.add_argument(
        '--simpledp-estimator-hyperparams', dest='ESTIMATOR_HYPERPARAMS',
        default=ConfigManager().get('ESTIMATOR_HYPERPARAMS',default=None),
        help=(
            'JSON text used as hyperparameters for the underlying algorithm. '
            'Together, estimator params and hyperparams are passed to the '
            'underlying algorithm, but hyperparams are usually subject to '
            'optimization. Overwrites ESTIMATOR_HYPERPARAMS env.'
        )
    )

    parser.add_argument(
        '--simpledp-estimator-hyperparams-search-space',
        dest='ESTIMATOR_HYPERPARAMS_SEARCH_SPACE',
        default=ConfigManager().get('ESTIMATOR_HYPERPARAMS_SEARCH_SPACE',default=None),
        help=(
            'JSON text of algorithm hyperparams range instrumented by the '
            'hyperparam optimizer. Overwrites '
            'ESTIMATOR_HYPERPARAMS_SEARCH_SPACE env.'
        )
    )



    parser.add_argument(
        '--debug', dest='DEBUG',
        action=argparse.BooleanOptionalAction,
        default=ConfigManager().get('DEBUG',default=False, cast=bool),
        help='Be more verbose and output messages to console.'
    )

    parsed = parser.parse_args()

    unset_if_none=[
        'DATAPROVIDER_LIST',             'XINGU_DB_TABLE_PREFIX',
        'QUERY_CACHE_PATH',              'DVC_QUERY_CACHE_PATH',
        'DVC_TRAINED_MODELS_PATH',       'PARALLEL_TRAIN_MAX_WORKERS',
        'HYPEROPT_STRATEGY',             'TRAINED_MODELS_PATH',
        'PRE_REQ_TRAIN_OR_SESSION_IDS',  'DATAPROVIDER_FOLDER',
        'HOSTNAME',                      'USERNAME',

        # Simple DataProvider configs
        'SIMPLEDP_ID',                   'BASE_CLASS',
        'TRAIN_DATASOURCE',              'TARGET_FEATURE',
        'ESTIMATOR_CLASS',               'ESTIMATOR_FEATURES',
        'PROBA_CLASS_INDEX',             'ESTIMATOR_CLASS_PARAMS',
        'ESTIMATOR_PARAMS',              'ESTIMATOR_HYPERPARAMS_SEARCH_SPACE',
        'ESTIMATOR_HYPERPARAMS'

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

    simpledp = None
    if 'SIMPLEDP_ID' in args:
        simpledp = DPSimple(**args)


    # Gather all DataProviders requested
    if 'DATAPROVIDER_LIST' in args:
        dpf = DataProviderFactory(
            providers_list=[x.strip() for x in args['DATAPROVIDER_LIST'].split(',')],
            providers_folder=args['DATAPROVIDER_FOLDER'],
            providers_extra_objects=simpledp,
        )
    else:
        dpf = DataProviderFactory(
            providers_folder=args['DATAPROVIDER_FOLDER'],
            providers_extra_objects=simpledp,
        )

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
        logger.info('Please use both or one of --batch-predict and --train')





if __name__ == "__main__":
    main()
