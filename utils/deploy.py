import os
import pathlib
import psutil
import s3path
import argparse
import logging
import shutil
import concurrent.futures
import sqlalchemy
import pandas as pd
import boto3


from .. import DataProviderFactory
from .. import Coach
from .. import Model
from .. import ConfigManager



def prepare_logging(level=logging.INFO):
    # Switch between INFO/DEBUG while running in production/developping:

    # Configure logging for Xingu

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
        prog='xingu.deploy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Copy Xingu DB Data and Trained PKLs between source and target environments.' + '\n' +
            'Usually, source is staging and target is production. But can be whatever you want, as:' + '\n' +
            '    production   → laptop (SQLite and local folder)' + '\n' +
            '    staging      → your SageMaker image (SQLite and local folder)' + '\n' +
            '    production   → your SageMaker image (SQLite and local folder)' + '\n' +
            '    a folder in your laptop (SQLite and local folder) → another folder in your laptop (SQLite and local folder)' + '\n' +
            '\n' +
            'Source and target are defined by following parameters or env:' + '\n' +
            '--source-xingu-db or SOURCE_XINGU_DB_URL with --target-xingu-db or TARGET_XINGU_DB_URL' + '\n' +
            '--source-trained-models-path or SOURCE_TRAINED_MODELS_PATH with --target-trained-models-path or TARGET_TRAINED_MODELS_PATH' + '\n' +
            '\n' +
            'Train IDs for each DataProvider that will be copied are defined by the currents.yaml file.'
    )

    parser.add_argument('--dps', '--data-providers', dest='DATAPROVIDER_LIST', required=False,
        default=ConfigManager.get('DATAPROVIDER_LIST', default=None),
        help='Comma separated list of DataProvider IDs to process. Overwrites DATAPROVIDER_LIST env. If not set, all classes inherited from DataProvider that have IDs will be used and a Xingu for each will be trained.')

    parser.add_argument('--db', dest='DO_DB', action=argparse.BooleanOptionalAction,
        required=False,
        default=ConfigManager.get('DO_DB',default=True, cast=bool),
        help='Copy database entries or not. Overwrites DO_DB env.')

    parser.add_argument('--db-page-size', dest='DB_PAGE_SIZE', type=int,
        default=ConfigManager.get('DB_PAGE_SIZE',default=0, cast=int),
        help='Table extraction in pages of this size to not exaust RAM. Default is unlimited. If RAM is 4GB, use 500000 here. Overwrites DB_PAGE_SIZE env.')

    parser.add_argument('--source-xingu-db', dest='SOURCE_XINGU_DB_URL', required=False,
        default=ConfigManager.get('SOURCE_xingu_DB_URL',default='sqlite:///xingu.db?check_same_thread=False'),
        help='URL for Xingu control database which contains data to be copied. Overwrites SOURCE_XINGU_DB_URL env. If empty, uses sqlite:///xingu.db?check_same_thread=False, which is a SQLite database on current folder.')

    parser.add_argument('--target-xingu-db', dest='TARGET_XINGU_DB_URL', required=False,
        default=ConfigManager.get('TARGET_XINGU_DB_URL',default='sqlite:///xingu.db?check_same_thread=False'),
        help='URL for Xingu control database which will receive data. Overwrites TARGET_XINGU_DB_URL env. If empty, uses sqlite:///xingu.db?check_same_thread=False, which is a SQLite database on current folder.')

    parser.add_argument('--source-table-prefix', dest='SOURCE_XINGU_DB_TABLE_PREFIX',
        default=ConfigManager.get('SOURCE_XINGU_DB_TABLE_PREFIX',default=None),
        help='A string to prefix every source table name with, such as “avi_”. Overwrites SOURCE_XINGU_DB_TABLE_PREFIX env.')

    parser.add_argument('--target-table-prefix', dest='TARGET_XINGU_DB_TABLE_PREFIX',
        default=ConfigManager.get('TARGET_XINGU_DB_TABLE_PREFIX',default=None),
        help='A string to prefix every source table name with, such as “avi_”. Overwrites TARGET_XINGU_DB_TABLE_PREFIX env.')

    parser.add_argument('--source-trained-models-path', dest='SOURCE_TRAINED_MODELS_PATH',
        default=ConfigManager.get('SOURCE_TRAINED_MODELS_PATH',default=None),
        help='A local folder or an S3 path to get trained models from. Example: «s3://loft-mlops-data/sample-dvc/teste-do-avi/avm-trained-models». Overwrites SOURCE_TRAINED_MODELS_PATH env.')

    parser.add_argument('--target-trained-models-path', dest='TARGET_TRAINED_MODELS_PATH',
        default=ConfigManager.get('TARGET_TRAINED_MODELS_PATH',default=None),
        help='A local folder or an S3 path to receive trained models from source. Overwrites TARGET_TRAINED_MODELS_PATH env.')

    parser.add_argument('--project-home', dest='PROJECT_HOME',
        default=ConfigManager.get('PROJECT_HOME',default='.'),
        help='Local clone of xingu_avm git repository, to collect various metadata during runtime. Overwrites PROJECT_HOME env.')


    parser.add_argument('--debug', dest='DEBUG', action=argparse.BooleanOptionalAction,
        default=ConfigManager.get('DEBUG',default=False, cast=bool),
        help='Be more verbose and output messages to console.')

    parsed = parser.parse_args()

    unset_if_none=[
        'DATAPROVIDER_LIST',
        'SOURCE_XINGU_DB_URL',                'TARGET_XINGU_DB_URL',
        'SOURCE_XINGU_DB_TABLE_PREFIX',       'TARGET_XINGU_DB_TABLE_PREFIX',
        'SOURCE_TRAINED_MODELS_PATH',          'TARGET_TRAINED_MODELS_PATH'
    ]

    args={}
    for c in parsed.__dict__:
        if parsed.__dict__[c] is not None or c not in unset_if_none:
            args.update({c: parsed.__dict__[c]})

    return args



def env_multiplexer(use, args):
    env=[
        'XINGU_DB_URL',
        'XINGU_DB_TABLE_PREFIX',
        'TRAINED_MODELS_PATH',
    ]
    
    multiplexed_env={}
    for e in env:
        if f'{use.upper()}_{e}' in args:
            if e in ConfigManager.cache:
                del ConfigManager.cache[e]
            multiplexed_env[e]=args[f'{use.upper()}_{e}']
    
    ConfigManager.set(multiplexed_env)
    


def copy_table(table, source_coach, target_coach):
    """
    Copy data of a single Xingu DB table related to requested DataProviders
    with the train_ids found in currents.yaml.
    """
    logger.info(f'Started copy of {table} table')

    deploy_map=dp_to_train_map(target_coach)

    orm_table=source_coach.tables[table]
    
    subfilter=[
        sqlalchemy.and_(
            orm_table.c.dataprovider_id == dp,
            orm_table.c.train_id == deploy_map[dp]
        )
        for dp in deploy_map
    ]
    
    query=orm_table.select().where(sqlalchemy.or_(*subfilter))
    
    if 'time' in orm_table.c:
        query=query.order_by(orm_table.c.time)
    
    # We’ll work in pages/chunks to avoid reaching RAM limits of machine
    if ConfigManager.get('DB_PAGE_SIZE', cast=int)>0:
        page_size=ConfigManager.get('DB_PAGE_SIZE')
    else:
        # Use a simple linear model to compute page size based on machine RAM
        ## 4GB RAM support around 450K rows
        ram1         =    4   *1000*1000*1000
        page_size1   =  400   *1000

        ## 64GB RAM support around 8M rows
        ram2         =   64   *1000*1000*1000
        page_size2   = 8000   *1000

        slope=(page_size2-page_size1)/(ram2-ram1)
        intercept=page_size1-slope*ram1

        # Find page size as a function of your RAM
        page_size=intercept+slope*psutil.virtual_memory().total
    
    page=0
    total=0
    
    while True:
        logger.debug(
            query.limit(page_size).offset(page*page_size).compile(
                compile_kwargs={'literal_binds': True},
            ).string
        )

        df=pd.read_sql(
            query.limit(page_size).offset(page*page_size),
            con=source_coach.get_db_connection('xingu')
        )
        
        total+=df.shape[0]
        
        logger.info(f'Table «{target_coach.tables[table]}», page {page} has {df.memory_usage(index=True).sum()} bytes in {df.shape[0]} lines')
        
#         logger.debug(df)
        
        df.to_sql(
            name=str(target_coach.tables[table]),
            index=False,
            if_exists='append',
            con=target_coach.get_db_connection('xingu')
        )
        
        if df.shape[0]==page_size:
            # current page is the maximum size, so probably there is more
            page+=1
        else:
            # current page is smaler than maximum size -> stop
            break
        
        # Tell garbage collector to free some RAM
        del df

    logger.info(f'Finished copy of {table} table')


    
def copy_tables(source_coach, target_coach):
    """
    Copy in parallel all Xingu DB data related to requested DataProviders
    with the train_ids found in currents.yaml.
    """
    main_table='training'
    
    # Main table must be the first because of foreign key reference constraints
    copy_table(main_table, source_coach, target_coach)
    
    with concurrent.futures.ThreadPoolExecutor(thread_name_prefix='copy_tables_parallel') as executor:
        tasks=[]
        
        # Submit all tasks
        for table in target_coach.tables:
            if table != main_table:
                tasks.append(executor.submit(copy_table, table, source_coach, target_coach))
        
        # Wait until all tasks finish
        for task in concurrent.futures.as_completed(tasks):
            task.result()



def copy_pkl(dataprovider_id, train_id, source_path, target_path):
    """
    Copy a single PKL for dataprovider_id matching train_id.
    """
    filename_template='{dataprovider_id} • {time} • {train_session_id}∶{train_id}.pkl*'

    filename=filename_template.format(
        dataprovider_id    = dataprovider_id,
        time               = '*',
        train_session_id   = '*',
        train_id           = train_id
    )

    # Get list of files in the path that match our filename glob
    # print(f'Searching for {resolved_path/filename}')
    s3path_bug_101=True # https://github.com/liormizr/s3path/issues/101
    if type(source_path)==s3path.S3Path and s3path_bug_101:
        # If s3path module still has the bug #101, we'll use s3fs to find
        # matches for the files we need.
        # When bug is resolved, we can use the more generic PathLib method
        # inside `else`.
        import s3fs
        s3=s3fs.S3FileSystem()
        available=[
            source_path / pathlib.PurePath(f).relative_to(str(source_path)[1:])
            for f in s3.glob(str(source_path / filename)[1:]) if '.dvc' not in str(f)
        ]
    else:
        available=[f for f in list(source_path.glob(filename)) if '.dvc' not in str(f)]
    
    if len(available)<1:
        raise FileNotFoundError(f'No estimator matches «{filename}»')
    
    available.sort()

    source=available[-1]
    target=target_path / source.name
    
    s3=None

    if type(source)==s3path.S3Path:
        if type(target)==s3path.S3Path:
            # This is a transfer from S3 to S3
            logger.info(f'Started S3 to S3 transfer of PKL for dataprovider={dataprovider_id} and train_id={train_id}')

            s3_source = dict(
                Bucket = source.bucket,
                Key = source.key
            )

            s3 = boto3.resource('s3')
            s3.meta.client.copy(s3_source, target.bucket, target.key)
        else:
            # This is a plain download from S3
            logger.info(f'Started download from S3 of PKL for dataprovider={dataprovider_id} and train_id={train_id}')
#             config = TransferConfig(use_threads=False)
            s3 = boto3.client('s3')
            s3.download_file(source.bucket, source.key, str(target)) #,Config=config)
    elif type(target)==s3path.S3Path:
        # This is a plain upload to S3
        logger.info(f'Started upload to S3 of PKL for dataprovider={dataprovider_id} and train_id={train_id}')
        s3 = boto3.client('s3')
        s3.upload_file(str(source), target.bucket, target.key)
    else:
        # This is a plain copy between regular filesystems, no S3
        logger.info(f'Started plain filesystem copy of PKL for dataprovider={dataprovider_id} and train_id={train_id}')
        shutil.copyfile(source,target)

    del s3
    
    logger.debug(f'Finished copy of PKL for dataprovider={dataprovider_id} and train_id={train_id}')



def copy_pkls(source_coach, target_coach):
    """
    Copy in parallel PKLs for all DataProviders requested with the train_ids
    found in currents.yaml.
    """
    locations=dict(
        source=ConfigManager.get('SOURCE_TRAINED_MODELS_PATH'),
        target=ConfigManager.get('TARGET_TRAINED_MODELS_PATH')
    )
    
    for l in locations:
        if locations[l].startswith('s3://'):
            locations[l]=s3path.S3Path.from_uri(locations[l])
        else:
            locations[l]=pathlib.Path(locations[l]).resolve()

    deploy_map=dp_to_train_map(target_coach)
            
    with concurrent.futures.ThreadPoolExecutor(thread_name_prefix='copy_pkls_parallel') as executor:
        tasks=[]
        for dp in deploy_map:
            params=dict(
                dataprovider_id   = dp,
                train_id          = deploy_map[dp],
                source_path       = locations['source'],
                target_path       = locations['target']
            )
            tasks.append(executor.submit(copy_pkl,**params))
        
        for task in concurrent.futures.as_completed(tasks):
            task.result()


            
def dp_to_train_map(target_coach: Coach) -> dict:
    """
    Returns a dict associating DataProvider ID (the key) with a train
    ID (the value).
    """
    deploy_map={}
    
    for dp in list(target_coach.dp_factory.produce()):
        if dp.id not in target_coach.currents['estimators']:
            logger.warning(f'{dp.id} not available in currents, but it was requested to be deployed.')
        else:
            deploy_map[dp.id]=target_coach.currents['estimators'][dp.id]['train_id']
    
    return deploy_map



def main():
    # Read environment and command line parameters
    args=prepare_args()


    # Setup logging
    global logger
    if args['DEBUG']:
        logger=prepare_logging(logging.DEBUG)
    else:
        logger=prepare_logging()

    if args['DO_DB']==False:
        # Adjust for no database operations
        args['SOURCE_XINGU_DB_URL']='sqlite:///___xingu.db'
        args['TARGET_XINGU_DB_URL']=args['SOURCE_XINGU_DB_URL']

    # Gather all DataProviders requested
    if 'DATAPROVIDER_LIST' in args:
        dpf=DataProviderFactory(providers_list=[x.strip() for x in args['DATAPROVIDER_LIST'].split(',')])
    else:
        dpf=DataProviderFactory()

    # Now set the environment with everything useful that we got
    ConfigManager.set(args)
        
    # Setup 2 Coach, for source and target environments.
    # Target has the filters in the dpf object.
    env_multiplexer('source', args)
    source_coach=Coach()
#     source_coach.get_db_connection('xingu')
    
    env_multiplexer('target', args)
    target_coach=Coach(dp_factory = dpf)
#     target_coach.get_db_connection('xingu')

    with concurrent.futures.ThreadPoolExecutor(thread_name_prefix='deploy_parallel') as executor:
        tasks=[]
        tasks.append(executor.submit(copy_pkls,    source_coach, target_coach))
        
        if args['DO_DB']==True:
            tasks.append(executor.submit(copy_tables,  source_coach, target_coach))

        for task in concurrent.futures.as_completed(tasks):
            task.result()



if __name__ == "__main__":
    main()
