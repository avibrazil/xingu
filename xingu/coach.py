import collections
import concurrent.futures
import copy
import datetime
import urllib
import importlib
import logging
import socket
import pathlib
import queue
import random
import os

import pandas
import yaml

from . import DataProvider
from . import DataProviderFactory
from . import Estimator
from . import ConfigManager



class Coach:

    ######################################################################################
    ##
    ## Initialization and configuration
    ##
    ######################################################################################


    defaults=dict(
        HYPEROPT_STRATEGY       = 'last',
        TRAINED_MODELS_PATH     = None,
        DVC_TRAINED_MODELS_PATH = None,
        QUERY_CACHE_PATH        = None,
        DVC_QUERY_CACHE_PATH    = None,
        XINGU_DB_TABLE_PREFIX  = '',
        DATA_DB_URL             = None,
        PROJECT_HOME            = '.',
    )



    def __init__(self, dp_factory: DataProviderFactory = DataProviderFactory()):
        self.config=ConfigManager()

        # Setup logging
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        # Turn attribute `databases` into a private copy
        self.databases=dict()

        try:
            import pygit2
            self.git_repo=pygit2.Repository(self.get_config('PROJECT_HOME'))
            self.logger.debug("Using git repo on {}".format(self.git_repo))
        except:
            self.git_repo=None

        self.dp_factory=dp_factory

        # Load our map of DataProvider IDs and their current train_ids
        self.load_inventory()

        # Database initialization block
        self.tables={}
        self.xingu_db=None

        # Flow control on queues and parallelism
        self.post_processing=False



    def get_config(self, config_item: str, default=ConfigManager.undefined, cast=ConfigManager.undefined):
        if default==ConfigManager.undefined:
            if config_item in self.defaults:
                return self.config.get(config_item, default=self.defaults[config_item], cast=cast)
            else:
                return self.config.get(config_item, cast=cast)
        else:
            return self.config.get(config_item, default=default, cast=cast)



    ######################################################################################
    ##
    ## Team train and team batch predict
    ##
    ## Methods are called following this topology:
    ##
    ## team_train():
    ##   - team_train_parallel() (background, parallelism controled by PARALLEL_TRAIN_MAX_WORKERS):
    ##     - team_load() (for pre-req models not trained in this session)
    ##     - Per DataProvider requested to be trained:
    ##       - team_train_member() (background):
    ##         - Model.fit()
    ##   - post_train_parallel() (background, only if POST_PROCESS=true):
    ##     - Per trained estimator (parallelism controled by PARALLEL_POST_PROCESS_MAX_WORKERS):
    ##       - Model.save() (background)
    ##       - Model.trainsets_save() (background)
    ##       - single_batch_predict() (background, only if bacth_predict=True):
    ##         - Model.compute_and_save_metrics()
    ##         - Model.save_batch_predict_estimations()
    ##   - update_inventory()
    ##
    ##
    ##
    ##
    ## If in batch-predict-only mode (use pre-trained estimators):
    ##
    ## team_batch_predict():
    ##   - team_load() (for all requested DPs)
    ##   - Per loaded model:
    ##     - single_batch_predict() (background):
    ##       - Model.compute_and_save_metrics()
    ##       - Model.save_batch_predict_estimations()
    ##
    ##
    ######################################################################################



    def trained_ids(self):
        """
        Return a set of just the IDs of what we have in our inventory of trained Models
        """
        return self.trained.keys()



    def team_train_member(self,dp):
        """
        Train a single Robson for DataProvider dp and add it to inventory
        """
        # Imported here to avoid circular dependency problems
        from xingu import Model

        # Actual model training
        model=Model(
            dp                   = dp,
            coach                = self,
            trained              = False,
            hyperopt_strategy    = self.get_config('HYPEROPT_STRATEGY')
        )

        self.logger.info(f'Start training {model.dp.id}')
        model.fit() # a lengthy process
        self.logger.info(f'Finished training {model.dp.id}')

        # Add to inventory of trained models
        self.declare_trained(model)



    def team_train_parallel(self):
        import randomname

        self.train_session_id=randomname.get_name()

        self.trained = {}
        self.trained_in_session = []

        self.logger.info('Training Models for the following DataProviders: ' + str(self.dp_factory.providers_list))


        ##################################################################################
        ##
        ## Compute which pre-trained models need to be loaded in order to train what
        ## we were requested to train. Load pre-trained pre-reqs with team_load()
        ##
        ##################################################################################

        to_train = list(self.dp_factory.produce())

        to_load=set()
        for dp in to_train:
            to_load = to_load | self.dp_factory.get_pre_req(dp)

        # Compute list of pre-req models that need to be loaded except the ones
        # that are going to be trained
        to_load = to_load - set([dp.id for dp in to_train])

        if len(to_load):
            # Check if we have special order requirements for pre-reqs
            pre_req_train_or_session_ids=self.get_config('PRE_REQ_TRAIN_OR_SESSION_IDS', default=None)
            if pre_req_train_or_session_ids is not None:
                # Convert to a list
                pre_req_train_or_session_ids=[x.strip() for x in pre_req_train_or_session_ids.split(',')]

            self.logger.info('Loading pre-trained models as pre-reqs: ' + str(to_load))
            self.team_load(
                to_load,
                pre_req_train_or_session_ids=pre_req_train_or_session_ids,
                post_process=False
            )

        random.shuffle(to_train)
        to_train=collections.deque(to_train)

        # Define how many parallel workers to execute
        max_workers=self.get_config('PARALLEL_TRAIN_MAX_WORKERS', default=0, cast=int)
        if max_workers == '' or max_workers == 0:
            max_workers=None
            self.logger.info(f'Training all possible Models in parallel')
        else:
            self.logger.info(f'Training {max_workers} Models in parallel')


        with concurrent.futures.ThreadPoolExecutor(
                            thread_name_prefix='team_train_parallel',
                            max_workers=max_workers
                ) as executor:
            processing=[]
            waiting=[]

            while len(to_train):
                # First try to submit all
                while len(to_train):
                    dp=to_train.popleft()
                    if hasattr(dp,'pre_req'):
                        if len(dp.pre_req) == len(dp.pre_req.intersection(self.trained_ids())):
                            self.logger.info('Trigger train of «{}»'.format(dp.id))
                            processing.append(executor.submit(self.team_train_member,dp))
                        else:
                            # Not ready for you yet. Go back to the queue.
                            self.logger.info('Deferring train of «{}» until all its pre-reqs are ready'.format(dp.id))
                            waiting.append(dp)
                    else:
                        # Model has no dependencies, so just train it
                        self.logger.info('Trigger train of «{}»'.format(dp.id))
                        processing.append(executor.submit(self.team_train_member,dp))

                # Requeue everything that we couldn't process due to pre-reqs
                to_train.extend(waiting)
                waiting=[]

                # Now block until something finishes
                tasks=concurrent.futures.wait(processing, return_when=concurrent.futures.FIRST_COMPLETED)
                for task in tasks.done:
                    # Let finished tasks express themselves. This is where in-task
                    # exceptions are raised.
                    e=task.exception()
                    processing.remove(task)
                    if e is None:
                        task.result()
                    else:
                        self.logger.warning('Exception occurred in team_train_parallel() tasks.')
                        for t in tasks.not_done:
                            # Cancel everything that we can cancel
                            t.cancel()
                        self.logger.exception(e)
                        raise e

        # Put an empty object to flag the end of queue
        if self.post_train_queue is not None:
            self.post_train_queue.put(None, block=False)



    def declare_trained(self, model, post_process=True):
        # Add to inventory of trained models
        self.trained[model.dp.id]=model

        # Add to inventory of models trained in this train session
        if hasattr(self,'trained_in_session'):
            self.trained_in_session.append(model.dp.id)

        if post_process and self.post_processing:
            # Put on a queue for post-process tasks, because
            # post_train_parallel() is waiting.
            self.post_train_queue.put(model, block=False)



    def team_train(self):
        """
        Train and post-process Models for all DPs requested.

        After an estimator is trained, it enters the post-process phase, which
        executes the following:
        1. Save PKL and commit to DVC
        2. Save Train, Validation and Test datasets to DB
        3. Get data and execute Batch Predict
        4. Save estimations in valuations DB table
        5. Compute various metrics and save them to DB.

        If POST_PROCESS env variable is set and is false, none of this will
        happen and this is usualy the behavior when we are only optimizing
        hyper-parameters.

        If POST_PROCESS env variable is not set or is true, post-processing
        activities are executed.

        If BATCH_PREDICT==False, steps from 3 and beyond are not executed.

        The BATCH_PREDICT flag is ignored if POST_PROCESS is false.
        """

        self.post_processing=False
        self.post_train_queue=None

        do_post_process=self.get_config('POST_PROCESS', default=True, cast=bool)

        with concurrent.futures.ThreadPoolExecutor(thread_name_prefix='team_train') as executor:
            # Only 2 tasks here:
            # - Trigger all trains in parallel, taking care of pre-reqs
            # - As soon as trained Models are ready, do their post processing
            tasks=[]

            # The parallel post-train task, including batch predict
            if do_post_process:
                # Initialize a queue for post-train activities
                self.post_train_queue=queue.Queue()

                # Let everybody know we are post-processing trained Models
                self.post_processing=True

                # Start post-processing
                tasks.append(executor.submit(self.post_train_parallel))

            # The parallel train task
            tasks.append(executor.submit(self.team_train_parallel))

            # Wait for both tasks (train and post-process) to finish
            for task in concurrent.futures.as_completed(tasks):
                # Let finished tasks express themselves. This is where in-task
                # exceptions are raised.
                e=task.exception()
                if e is None:
                    task.result()
                else:
                    self.logger.warning('Exception ocurred in team_train() tasks. Forcing a shutdown in post-process task.')
                    self.post_processing=False
                    if self.post_train_queue is not None:
                        self.post_train_queue.put(None, block=False)
                        self.logger.debug('Post process queue size: ' + str(self.post_train_queue.qsize()))
                    self.logger.exception(e)
                    raise e

            self.post_processing=False

        # Block until everything is done in this post-train queue
        if self.post_train_queue is not None:
            self.logger.info('Wait for post-process to finish while {} tasks in queue.'.format(self.post_train_queue.qsize()))
            self.post_train_queue.join()

        for model in self.trained_in_session:
            self.trained[model].cleanup()
            # If DVC is active, there will be things to commit
            self.trained[model].dvc_commit()

        self.update_inventory()



    def post_train_parallel(self):
        """
        Listens to trained Models and fire post-process tasks such as saving model,
        saving data, batch predict, metrics computation etc.
        """

        # Define how many parallel workers to execute
        max_workers=self.get_config('PARALLEL_POST_PROCESS_MAX_WORKERS', default=0, cast=int)
        if max_workers == '' or max_workers == 0:
            max_workers=None
            self.logger.info(f'Post-processing all possible Models in parallel')
        else:
            self.logger.info(f'Post-processing no more than {max_workers} Models in parallel')


        with concurrent.futures.ThreadPoolExecutor(
                            thread_name_prefix='post_train_parallel',
                            max_workers=max_workers
                ) as executor:
            tasks=[]
            while True:
                # Block until a Model object is trained and ready to be used for post processing
                model=self.post_train_queue.get()
                self.post_train_queue.task_done()

                self.logger.debug('Post process queue size after queue.get(): ' + str(self.post_train_queue.qsize()))

                if model is None:
                    # A sign that all was done. Exit the endless loop.
                    self.post_processing=False
                    self.logger.info('All post-process tasks done. Shutting down queue.')
                    break

                self.logger.info(f'A fresh Model({model.dp.id}) is ready for post-processing')


                # Now do all the amazing things we can do with a trained
                # Model. In parallel.

                ## Save pickle to storage
                tasks.append(
                    executor.submit(
                        model.save,
                        **dict(
                            path      = self.get_config(    'TRAINED_MODELS_PATH'),
                            dvc_path  = self.get_config('DVC_TRAINED_MODELS_PATH'),
                        )
                    )
                )

                # Save trainsets to DB
                tasks.append(executor.submit(model.trainsets_save))

                # Predict and compute metrics over train data
                tasks.append(executor.submit(model.trainsets_predict))

                # Batch predict and metrics computation
                if self.get_config('BATCH_PREDICT', default=False, cast=bool):
                    tasks.append(executor.submit(self.single_batch_predict,model))
                else:
                    self.logger.info('Skipping batch predict activities.')


            for task in concurrent.futures.as_completed(tasks):
                # Let finished tasks express themselves. This is where
                # in-task exceptions are raised.

                self.logger.debug('A post-process task is done.')

                e=task.exception()
                if e is None:
                    # No exception
                    task.result()
                else:
                    self.logger.warning('Exception ocurred in team_train() tasks. Forcing a shutdown in post-process task.')
                    self.post_processing=False
                    self.post_train_queue.put(None, block=False)
                    self.logger.exception(e)
                    raise e

            self.logger.debug('Alls post-process tasks finished.')



    def single_batch_predict(self, model):
        model.batch_predict()

        # For each trained Robson, do 2 tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(thread_name_prefix='single_batch_predict') as executor:
            tasks=[]

            # Task 1: compute and save metrics
            tasks.append(executor.submit(model.compute_and_save_metrics,'batch'))

            # Task 2: save estimations
            if self.get_config('BATCH_PREDICT_SAVE_ESTIMATIONS', default=False, cast=bool):
                tasks.append(executor.submit(model.save_batch_predict_estimations))

            for task in concurrent.futures.as_completed(tasks):
                # Does nothing if thread succeeded. Raises the task's exception otherwise.
                task.result()



    def team_batch_predict(self):
        if not hasattr(self, 'trained'):
            self.team_load()

        # Define how many parallel workers to execute
        max_workers=self.get_config('PARALLEL_POST_PROCESS_MAX_WORKERS', default=0, cast=int)
        if max_workers == '' or max_workers == 0:
            max_workers=None
            self.logger.info(f'Using all possible Models in parallel')
        else:
            self.logger.info(f'Using no more than {max_workers} Models in parallel')

        # All models in place, so now batch predict in parallel
        with concurrent.futures.ThreadPoolExecutor(
                            thread_name_prefix='team_batch_predict',
                            max_workers=max_workers
                ) as executor:
            tasks=[]

            for model in self.trained:
                tasks.append(executor.submit(self.single_batch_predict, self.trained[model]))

            for task in concurrent.futures.as_completed(tasks):
                # Does nothing if thread succeeded. Raises the task's exception otherwise.
                task.result()



    def update_inventory(self):
        if hasattr(self,'trained') and hasattr(self,'trained_in_session'):
            # Make any needed initialization
            if 'history' not in self.inventory:
                self.inventory['history']=[]

            if 'models' not in self.inventory:
                self.inventory['models']=dict()

            # Update file with what we have
            self.inventory['history'].append(
                dict(
                    time              = datetime.datetime.now(datetime.timezone.utc),
                    user_name         = os.environ.get('USER', os.environ.get('USERNAME')),
                    host_name         = socket.gethostname(),
                    git_branch        = self.git_repo.head.name if self.git_repo else None,
                    git_commit        = self.git_repo.head.target.hex if self.git_repo else None,
                    github_actor      = self.get_config('GITHUB_ACTOR', None),
                    github_workflow   = self.get_config('GITHUB_WORKFLOW', None),
                    github_run_id     = self.get_config('GITHUB_RUN_ID', None),
                    github_run_number = self.get_config('GITHUB_RUN_NUMBER', None),
                    trained_in_session= self.trained_in_session,
                    train_session_id  = self.train_session_id
                )
            )

            # Preserve just last 10 entries in history
            self.inventory['history']=self.inventory['history'][-10:]

            for e in self.trained_in_session:
                if e in self.inventory['models']:
                    del self.inventory['models'][e]
                self.inventory['models'][e]=self.trained[e].signature()

            target=pathlib.Path(self.get_config('PROJECT_HOME')).resolve() / 'inventory.yaml'
            with open(target, 'w') as f:
                yaml.dump(
                    data=self.inventory,
                    stream=f,

                    # YAML styling options
                    explicit_start=True,
                    canonical=False,
                    indent=6,
                    default_flow_style=False
                )

            self.git_commit_inventory()



    def load_inventory(self) -> dict:
        target=pathlib.Path(self.get_config('PROJECT_HOME')).resolve() / 'inventory.yaml'

        try:
            self.logger.debug("Trying to open {}".format(target))
            with open(target) as f:
                self.inventory = yaml.safe_load(f)
        except FileNotFoundError as e:
            self.inventory={}

        if self.inventory is None:
            self.inventory={}

        return self.inventory



    def git_commit_inventory(self):
        if self.get_config('COMMIT_INVENTORY', default=False, cast=bool):
            git_shell = """
                cd {path};
                git add 'inventory.yaml';
                git commit -m 'chore(inventory.yaml)';
                git push;
            """

            git_shell=git_shell.format(path=pathlib.Path(self.get_config('PROJECT_HOME')).resolve())
            self.logger.debug(f'Commit inventory.yaml to Git with command:\n{git_shell}')
            return_code=os.system(git_shell)

            if return_code != 0:
                raise OSError(return_code, f'Following external command failed: {git_shell}')


    ######################################################################################
    ##
    ## Services for team coordination
    ##
    ######################################################################################


    def find_model_for_dp(self, dp: str):
        if hasattr(self,'trained'):
            return self.trained[dp]



    def team_load(
                    self,
                    explicit_list                              = None,
                    post_process:                         bool = True,
                    estimator_only:                       bool = False,
                    pre_trained_path:                     str  = None,
                    pre_trained_train_session_id:         str  = '*',
                    pre_trained_as_of:                    str  = '*',
                    pre_req_train_or_session_ids:         list = None,
                    xingu_module:                         str  = 'xingu',
                    model_class:                          str  = 'Model'
        ):
        """
        Load pre-trained Xingu models into Coach’s inventory.

        PKLs will be searched in S3 or filesystem path defined in
        pre_trained_path.

        explicit_list -- Load estimators only for this DataProviders. Load all
        if empty or None.

        post_process -- If True, submit loaded models for post-processing
        (batch predict, metrics computation etc)
        """
        xingu_module = importlib.import_module(xingu_module)
        Model = getattr(xingu_module, model_class)

        model_params_template=dict(
            # Our optimization technique
            delayed_prereq_binding=               True,

            # Obvious parameter
            trained=                              True,

            # Other stuff
            estimator_only=                       estimator_only,
            pre_trained_path=                     pre_trained_path,
            pre_trained_train_session_id=         pre_trained_train_session_id,
            pre_trained_train_id=                 '*',
            pre_trained_as_of=                    pre_trained_as_of,
            train_or_session_id_match_list=       pre_req_train_or_session_ids
        )

        # Start with a fresh empty team
        self.trained = dict()

        if explicit_list is None:
            explicit_list=list(self.dp_factory.produce())

        # Find all pre-reqs of desired models and add them to the list
        to_load = set()
        for dp in explicit_list:
            if isinstance(dp, str):
                to_load=to_load | {dp}    | self.dp_factory.get_pre_req(dp)
            else:
                to_load=to_load | {dp.id} | self.dp_factory.get_pre_req(dp)

        self.logger.debug(f'Going to load pre-trained: {to_load}')

        # Parallel load of all desired pre-trained Robson objects
        with concurrent.futures.ThreadPoolExecutor(thread_name_prefix='team_load') as executor:
            tasks = []

            for dp in to_load:
                model_params = copy.deepcopy(model_params_template)
                model_params['coach'] = self
                model_params['dp'] = self.dp_factory.dps[dp]() if isinstance(dp, str) else dp

                if (
                            'models' in self.inventory and
                            dp in self.inventory['models']
                ):
                    model_params['pre_trained_train_id'] = self.inventory['models'][dp]['train_id']
                    model_params['train_or_session_id_match_list'] = None

                self.logger.info(f'Trigger background loading of ‘{dp}’')
                tasks.append(executor.submit(Model, **model_params))

            for task in concurrent.futures.as_completed(tasks):
                # Does nothing if thread succeeded. Raises the task's exception otherwise.
                mod = task.result()
                self.declare_trained(mod, post_process=post_process)
                self.logger.info(f'Loaded {mod}')

        # Now bind pre-req models with the ones already in the team.
        for model in self.trained:
            self.trained[model].load_pre_req_model()



    ######################################################################################
    ##
    ## Metrics report
    ##
    ######################################################################################


    def report(self, train_ids: list=None, dataprovider_id: str=None, on: str='model',
                    start: str=None, reference_train_id: str=None) -> pandas.DataFrame:
        
        import sqlalchemy

        # Method idealized but still unimplemented.

        metric_table            = self.tables['metrics_' + on]
        training_table          = self.tables['training']
        training_status_table   = self.tables['training_status']

        query_meta=(
            training_table.select()
                .join(training_status_table, sqlalchemy.and_(
                        training_status_table.c.dataprovider_id == training_table.c.dataprovider_id,
                        training_status_table.c.train_id        == training_table.c.train_id
                    )
                )
                .filter(training_status_table.c.status == 'train_fit_end')

                .with_only_columns(
                    training_status_table.c.time,
                    training_table.c.dataprovider_id,
                    training_table.c.train_session_id,
                    training_table.c.train_id,
                    training_table.c.user_name,
                    training_table.c.host_name,
                    training_table.c.git_branch,
                    training_table.c.git_commit,
                    training_status_table.c.hyperparam
                )
        )

        if on == 'model':
            column_set='time_utc dataprovider_id train_session_id train_id set name'.split()
            time_table=training_status_table

            query_metrics=(
                metric_table.select()
                    .join(training_status_table, sqlalchemy.and_(
                            training_status_table.c.dataprovider_id == metric_table.c.dataprovider_id,
                            training_status_table.c.train_id        == metric_table.c.train_id
                        )
                    )
                    .filter(training_status_table.c.status == 'model_metrics_end')

                    .with_only_columns(
#                         training_status_table.c.time,
#                         metric_table.c.dataprovider_id,
#                         metric_table.c.train_session_id,
                        metric_table.c.train_id,
                        metric_table.c.set,
                        metric_table.c.name,
                        metric_table.c.value_number,
                        metric_table.c.value_text,
                    )
            )

        elif on == 'valuation':
            column_set='unit_id dataprovider_id time_utc train_id name'.split()
            time_table=metric_table

            query_metrics=(
                metric_table.select()
                    .with_only_columns(
                        metric_table.c.unit_id,
                        metric_table.c.dataprovider_id,
                        metric_table.c.time,
                        metric_table.c.train_session_id,
                        metric_table.c.train_id,
                        metric_table.c.name,
                        metric_table.c.value_number,
                        metric_table.c.value_text,
                    )
            )


        if dataprovider_id is not None:
            query_metrics = query_metrics.filter(metric_table.c.dataprovider_id == dataprovider_id)
            query_meta    = query_meta.filter(training_table.c.dataprovider_id == dataprovider_id)

        all_train_ids=[]
        if isinstance(train_ids, list):
            all_train_ids=train_ids

        if reference_train_id is not None:
            all_train_ids.append(reference_train_id)

        if len(all_train_ids)>0:
            query_metrics = query_metrics.filter(metric_table.c.train_id.in_(all_train_ids))
            query_meta    = query_meta.filter(training_table.c.train_id.in_(all_train_ids))

        if start is not None:
            query_metrics = query_metrics.filter(time_table.c.time >= pd.Timestamp(start).timestamp())
            query_meta    = query_meta.filter(training_status_table.c.time >= pd.Timestamp(start).timestamp())


        # Now execute query
        report={}
        self.logger.debug(
            'Estimator metadata retrieval query: ' +
            query_meta.compile(compile_kwargs={'literal_binds': True}).string
        )
        self.logger.debug(
            'Metric retrieval query: ' +
            query_metrics.compile(compile_kwargs={'literal_binds': True}).string
        )
        report['meta']    = pd.read_sql(query_meta,    con=query_meta.bind)
        report['metrics'] = pd.read_sql(query_metrics, con=query_metrics.bind)

        # Handle time
        report['meta']['time_utc']=pd.to_datetime(report['meta']['time'], unit='s', utc=True)
        report['meta']=report['meta'].set_index('train_id').drop(columns=['time']).T

        report['metrics']['value'] = (
            report['metrics']['value_number']
            .combine_first(report['metrics']['value_text'])
        )

        report['metrics']=(
            report['metrics']
            .drop(columns=['value_number','value_text'])
            .set_index(['name','set','train_id'])
            .unstack(level=['train_id','set'])
        )


        return report

        df=df.set_index(column_set).sort_index().unstack()

        for c in df.columns:
            if df[c].value_counts().shape[0]==0:
                # Detected a column containing nothing but NaNs, a residual from unstack(); remove it
                df.drop(columns=[c], inplace=True)
        df.columns=df.columns.droplevel(0)

        df.sort_index(inplace=True)

        return df



    ######################################################################################
    ##
    ## Database management and creation
    ##
    ######################################################################################


    def get_db_connection(self, nickname='xingu'):
        if nickname in self.databases:
            return self.databases[nickname]['conn']

        import sqlalchemy

        engine_config_sets=dict(
            # Documentation for all these SQLAlchemy pool control parameters:
            # https://docs.sqlalchemy.org/en/14/core/engines.html#engine-creation-api

            DEFAULT=dict(
                # QueuePool config for a real database
                poolclass         = sqlalchemy.pool.QueuePool,

                # 5 is the default. Reinforce default, which is good
                pool_size         = 5,

                # Default here was 10, which might be low sometimes, so
                # increase to some big number in order to never let the
                # QueuePool be a bottleneck.
                max_overflow      = 50,

                # Debug connection and all queries
                # echo              = True
            ),
            sqlite=dict(
                # SQLite doesn’t support concurrent writes, so we‘ll amend
                # the DEFAULT configuration to make the pool work with only
                # 1 simultaneous connection. Since Robson is agressively
                # parallel and requires a DB service that can be used in
                # parallel (regular DBs), the simplicity and portability
                # offered by SQLite for a light developer laptop has its
                # tradeoffs and we’ll have to tweak it to make it usable in
                # a parallel environment even if SQLite is not parallel.

                # A pool_size of 1 allows only 1 simultaneous connection.
                pool_size         = 1,
                max_overflow      = 0,

                # Since we have only 1 stream of work (pool_size=1),
                # we need to put a hold on other DB requests that arrive
                # from other parallel tasks. We do this putting a high value
                # on pool_timeout, which controls the number of seconds to
                # wait before giving up on getting a connection from the
                # pool.
                pool_timeout      = 3600.0,

                # Debug connection and all queries
                # echo              = True
            ),
        )

        # Convert the text on environment ("nick1|url1|nick2|url2") into
        # a clean list
        databases=self.get_config('DATABASES', default=None)

        if databases is None:
            databases=list()
        else:
            databases=[i for i in [i.strip() for i in databases.split('|')] if i != '']

        # Process Xingu DB likewise with nickname 'xingu'
        if nickname == 'xingu':
            databases += ['xingu',self.get_config('XINGU_DB_URL')]

        if (len(databases) % 2) != 0:
            raise NotImplementedError('Malformed DATABASES environment. Format is "nick1|url1|nick2|url2".')

        for i in range(0,len(databases),2):
            self.databases[databases[i]] = dict(
                url=databases[i+1]
            )

            current=self.databases[databases[i]]

            if 'athena' in current['url']:
                # Set some defaults for AWS Athena in here to avoid global
                # module requirements
                import pyathena.pandas.cursor
                engine_config_sets['athena']=dict(
                    connect_args=dict(
                        cursor_class=pyathena.pandas.cursor.PandasCursor
                    )
                )

            # Start with a default config
            engine_config=engine_config_sets['DEFAULT'].copy()

            # Add engine-specific configs
            for dbtype in engine_config_sets.keys():
                # Extract from engine_config_sets configuration specific
                # for each DB type
                if dbtype in current['url']:
                    engine_config.update(engine_config_sets[dbtype])

            # Databricks needs special URL handling
            # URLs are like "databricks+connector://host.com/default?http_path=/sql/..."
            if 'databricks' in current['url']:
                tokenized = urllib.parse.urlparse(current['url'])

                # Extract connection args as "?http_path=..." into a dict
                # Returns {'http_path': ['value']}
                conn_args = urllib.parse.parse_qs(tokenized.query)

                # Unwrap the value from the list
                # Return  {'http_path': 'value'}
                engine_config.update(
                    dict(
                        connect_args={k:conn_args[k][0] for k in conn_args}
                    )
                )

                # Reconstruct the URL without the connection args
                tokenized=tokenized._replace(query=None)
                current['url']=urllib.parse.urlunparse(tokenized)

            current['conn'] = sqlalchemy.create_engine(
                url = current['url'],
                **engine_config
            )

            # DB Engine was just created; check if all tables are there
            if databases[i] == 'xingu':
                self.init_db(current['conn'])

            self.logger.debug(f"Data source «{databases[i]}» is {current['conn']}. Created with config: {engine_config}.")

        return self.databases[nickname]['conn']



    def init_db(self, con):
        if self.get_config('XINGU_DB_URL'):
            # This is just to raise an exception if not set.
            # Can't do Coach business without a DB.
            pass

        import sqlalchemy

        self.xingu_db_metadata = sqlalchemy.MetaData(bind=con)

        self.xingu_db_table_prefix=self.get_config('XINGU_DB_TABLE_PREFIX','')

        table_name='training'
        self.tables[table_name] = sqlalchemy.Table(
            self.xingu_db_table_prefix + table_name,
            self.xingu_db_metadata,
            sqlalchemy.Column(
                'train_session_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'train_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'dataprovider_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),

            sqlalchemy.UniqueConstraint(
                'train_session_id',
                'train_id',
                'dataprovider_id',
                name='train'
            )
        )

        table_name='training_attributes'
        self.tables[table_name] = sqlalchemy.Table(
            self.xingu_db_table_prefix + table_name,
            self.xingu_db_metadata,
            sqlalchemy.Column(
                'train_session_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'train_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'dataprovider_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'attribute', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'value', sqlalchemy.String
            ),

            sqlalchemy.UniqueConstraint(
                'train_session_id',
                'train_id',
                'dataprovider_id',
                'attribute',
                name='train'
            )
        )

        table_name='training_steps'
        self.tables[table_name] = sqlalchemy.Table(
            self.xingu_db_table_prefix + table_name,
            self.xingu_db_metadata,
            sqlalchemy.Column(
                'time', sqlalchemy.Integer,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'dataprovider_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'train_session_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'train_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'status', sqlalchemy.String,
                primary_key=True
            ),

            sqlalchemy.ForeignKeyConstraint(
                [
                    'dataprovider_id',
                    'train_session_id',
                    'train_id'
                ],
                [
                    self.xingu_db_table_prefix + 'training.dataprovider_id',
                    self.xingu_db_table_prefix + 'training.train_session_id',
                    self.xingu_db_table_prefix + 'training.train_id',
                ]
            ),

            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_time', 'time'),
            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_dataprovider_id', 'dataprovider_id'),
            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_train_session_id', 'train_session_id'),
            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_train_id', 'train_id'),
            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_status', 'status'),
        )

        table_name='sets'
        self.tables[table_name] = sqlalchemy.Table(
            self.xingu_db_table_prefix + table_name,
            self.xingu_db_metadata,
            sqlalchemy.Column(
                'dataprovider_id', sqlalchemy.String,
                nullable=False
            ),
            sqlalchemy.Column(
                'train_session_id', sqlalchemy.String,
                nullable=False
            ),
            sqlalchemy.Column(
                'train_id', sqlalchemy.String,
                nullable=False
            ),
            sqlalchemy.Column(
                'set', sqlalchemy.String,
                nullable=False
            ),
            sqlalchemy.Column(
                'index', sqlalchemy.String,
                nullable=False
            ),
            sqlalchemy.Column('target', sqlalchemy.Float),

            sqlalchemy.ForeignKeyConstraint(
                [
                    'dataprovider_id',
                    'train_session_id',
                    'train_id'
                ],
                [
                    self.xingu_db_table_prefix + 'training.dataprovider_id',
                    self.xingu_db_table_prefix + 'training.train_session_id',
                    self.xingu_db_table_prefix + 'training.train_id',
                ]
            ),

            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_dataprovider_id', 'dataprovider_id'),
            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_train_session_id', 'train_session_id'),
            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_train_id', 'train_id'),
        )

        table_name='metrics_model'
        self.tables[table_name] = sqlalchemy.Table(
            self.xingu_db_table_prefix + table_name,
            self.xingu_db_metadata,
            sqlalchemy.Column(
                'time', sqlalchemy.Integer,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'dataprovider_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'train_session_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'train_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'set', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'name', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column('value_number', sqlalchemy.Float),
            sqlalchemy.Column('value_text', sqlalchemy.String),

            sqlalchemy.ForeignKeyConstraint(
                [
                    'dataprovider_id',
                    'train_session_id',
                    'train_id'
                ],
                [
                    self.xingu_db_table_prefix + 'training.dataprovider_id',
                    self.xingu_db_table_prefix + 'training.train_session_id',
                    self.xingu_db_table_prefix + 'training.train_id',
                ]
            ),

            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_time', 'time'),
            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_dataprovider_id', 'dataprovider_id'),
            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_train_session_id', 'train_session_id'),
            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_train_id', 'train_id'),
        )

        table_name='metrics_estimation'
        self.tables[table_name] = sqlalchemy.Table(
            self.xingu_db_table_prefix + table_name,
            self.xingu_db_metadata,
            sqlalchemy.Column(
                'time', sqlalchemy.Integer,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'train_session_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'train_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'dataprovider_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'index', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'name', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column('value_number', sqlalchemy.Float),
            sqlalchemy.Column('value_text', sqlalchemy.String),

            sqlalchemy.ForeignKeyConstraint(
                [
                    'dataprovider_id',
                    'train_session_id',
                    'train_id'
                ],
                [
                    self.xingu_db_table_prefix + 'training.dataprovider_id',
                    self.xingu_db_table_prefix + 'training.train_session_id',
                    self.xingu_db_table_prefix + 'training.train_id',
                ]
            ),

            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_time', 'time'),
            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_dataprovider_id', 'dataprovider_id'),
            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_train_session_id', 'train_session_id'),
            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_train_id', 'train_id'),
        )

        table_name='estimations'
        self.tables[table_name] = sqlalchemy.Table(
            self.xingu_db_table_prefix + table_name,
            self.xingu_db_metadata,
            sqlalchemy.Column(
                'time', sqlalchemy.Integer,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'dataprovider_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'train_session_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'train_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column(
                'index', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            sqlalchemy.Column('estimation', sqlalchemy.Float),

            sqlalchemy.ForeignKeyConstraint(
                [
                    'dataprovider_id',
                    'train_session_id',
                    'train_id'
                ],
                [
                    self.xingu_db_table_prefix + 'training.dataprovider_id',
                    self.xingu_db_table_prefix + 'training.train_session_id',
                    self.xingu_db_table_prefix + 'training.train_id',
                ]
            ),

            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_time', 'time'),
            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_dataprovider_id', 'dataprovider_id'),
            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_train_session_id', 'train_session_id'),
            sqlalchemy.Index(self.xingu_db_table_prefix + table_name + '_by_train_id', 'train_id'),
        )

        self.logger.info('Going to create tables on Xingu DB')

        self.xingu_db_metadata.create_all()