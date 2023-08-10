import collections
import concurrent.futures
import copy
import datetime
import urllib
import importlib
import logging
import socket
import pathlib
import pandas
import pygit2
import pwd
import queue
import random
import randomname
import sqlalchemy
import os
import yaml
# import sqlalchemy.pool
from sqlalchemy import (Column, Integer, String, Float, DateTime, Table,
    MetaData, Index, ForeignKeyConstraint, UniqueConstraint)  # , JSON, ARRAY
from . import DataProvider
from . import DataProviderFactory
from . import panestimator
from . import NGBClassic
from . import PanConfigManager



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
        MODELS_DB_TABLE_PREFIX  = '',
        UNITS_DB_URL            = None,
        PROJECT_HOME            = '.',
    )


    databases=dict(
        robson=dict(
            env="MODELS_DB_URL",
        ),
        datalake_athena=dict(
            env="DATALAKE_ATHENA_URL",
        ),
        datalake_databricks=dict(
            env="DATALAKE_DATABRICKS_URL",
        ),
    )
    

    def __init__(self, dp_factory: DataProviderFactory = DataProviderFactory()):
        # Setup logging
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        # Turn attribute `databases` into a private copy
        self.databases=copy.deepcopy(self.databases)
        
        try:
            self.git_repo=pygit2.Repository(self.get_config('PROJECT_HOME'))
            self.logger.debug("Using git repo on {}".format(self.git_repo))
        except:
            self.git_repo=None

        self.dp_factory=dp_factory

        # Load our map of DataProvider IDs and their current train_ids
        self.load_currents_file()
        
        # Database initialization block
        self.tables={}
        self.panmodels_db=None
        self.credit_db=None
        self.get_db_connection('panmodels')
#         self.get_units_db()
        self.init_db()

        # Flow control on queues and parallelism
        self.post_processing=False



    def get_config(self, config_item: str, default=PanConfigManager.undefined, cast=PanConfigManager.undefined):
#         if 'ROBSON_DB' == config_item:
#             return self.get_robson_db()

#         elif 'UNITS_DB' == config_item:
#             return self.get_units_db()

#         else:
        if default==PanConfigManager.undefined:
            if config_item in self.defaults:
                return PanConfigManager.get(config_item, default=self.defaults[config_item], cast=cast)
            else:
                return PanConfigManager.get(config_item, cast=cast)
        else:
            return PanConfigManager.get(config_item, default=default, cast=cast)



    ######################################################################################
    ##
    ## Team train and team batch predict
    ##
    ## Methods are called following this topology:
    ##
    ## team_train():
    ##   - team_train_parallel() (background, parallelism controled by PARALLEL_TRAIN_MAX_WORKERS):
    ##     - team_load() (for pre-req estimators not trained in this session)
    ##     - Per DataProvider requested to be trained:
    ##       - team_train_member() (background):
    ##         - Robson.fit()
    ##   - post_train_parallel() (background, only if POST_PROCESS=true):
    ##     - Per trained estimator (parallelism controled by PARALLEL_POST_PROCESS_MAX_WORKERS):
    ##       - Robson.save() (background)
    ##       - Robson.save_train_sets() (background)
    ##       - single_batch_predict() (background, only if bacth_predict=True):
    ##         - Robson.compute_and_save_metrics()
    ##         - Robson.save_batch_predict_valuations()
    ##   - update_currents_file()
    ##
    ##
    ##
    ##
    ## If in batch-predict-only mode (use pre-trained estimators):
    ##
    ## team_batch_predict():
    ##   - team_load() (for all requested DPs)
    ##   - Per loaded estimator:
    ##     - single_batch_predict() (background):
    ##       - Robson.compute_and_save_metrics()
    ##       - Robson.save_batch_predict_valuations()
    ##
    ##
    ######################################################################################



    def trained_ids(self):
        """
        Return a set of just the IDs of what we have in our inventory of trained Robsons
        """
        return self.trained.keys()



    def team_train_member(self,dp):
        """
        Train a single Robson for DataProvider dp and add it to inventory
        """
        # Imported here to avoid circular dependency problems
        from robson import Robson

        # Actual model training
        model=Robson(
            dp                   = dp,
            coach                = self,
            estimator_class      = NGBClassic,
            trained              = False,
            hyperopt_strategy    = self.get_config('HYPEROPT_STRATEGY')
        )

        self.logger.info(f'Start training {model.dp.id}')
        model.fit() # a lengthy process
        self.logger.info(f'Finished training {model.dp.id}')

        # Add to inventory of trained models
        self.declare_trained(model)



    def team_train_parallel(self):
        self.train_session_id=randomname.get_name()

        self.trained={}
        self.trained_in_session=[]

        self.logger.info('Training Robsons for the following DataProviders: ' + str(self.dp_factory.providers_list))


        ##################################################################################
        ##
        ## Compute which pre-trained models need to be loaded in order to train what
        ## we were requested to train. Load pre-trained pre-reqs with team_load()
        ##
        ##################################################################################

        to_train=list(self.dp_factory.produce())

        to_load=set()
        for dp in to_train:
            to_load=to_load | self.dp_factory.get_pre_req(dp)

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
            self.logger.info(f'Training all possible Robsons in parallel')
        else:
            self.logger.info(f'Training {max_workers} Robsons in parallel')


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
                            self.logger.info('Deferring train of «{}» until all pre-reqs are ready'.format(dp.id))
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



    def team_train(self, batch_predict=True):
        """
        Train and post-process Robsons for all DPs requested.
        
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
        
        If batch_predict==False, steps from 3 and beyond are not executed.
        
        The batch_predict flag is ignored if POST_PROCESS is false.
        """

        self.post_processing=False
        self.post_train_queue=None

        do_post_process=self.get_config('POST_PROCESS', default=True, cast=bool)

        with concurrent.futures.ThreadPoolExecutor(thread_name_prefix='team_train') as executor:
            # Only 2 tasks here:
            # - Trigger all trains in parallel, taking care of pre-reqs
            # - As soon as trained Robons are ready, do their post processing
            tasks=[]

            # The parallel post-train task, including batch predict
            if do_post_process:
                # Initialize a queue for post-train activities
                self.post_train_queue=queue.Queue()

                # Let everybody know we are post-processing trained Robsons
                self.post_processing=True

                # Start post-processing
                tasks.append(executor.submit(self.post_train_parallel, batch_predict))

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
        
        self.update_currents_file()



    def post_train_parallel(self, batch_predict):
        """
        Listens to trained Robsons and fire post-process tasks such as saving model,
        saving data, batch predict, metrics computation etc.
        """

        # Define how many parallel workers to execute
        max_workers=self.get_config('PARALLEL_POST_PROCESS_MAX_WORKERS', default=0, cast=int)
        if max_workers == '' or max_workers == 0:
            max_workers=None
            self.logger.info(f'Post-processing all possible Robsons in parallel')
        else:
            self.logger.info(f'Post-processing no more than {max_workers} Robsons in parallel')


        with concurrent.futures.ThreadPoolExecutor(
                            thread_name_prefix='post_train_parallel',
                            max_workers=max_workers
                ) as executor:
            tasks=[]
            while True:
                # Block until a Robson object is trained and ready to be used for post processing
                model=self.post_train_queue.get()
                self.post_train_queue.task_done()

                self.logger.debug('Post process queue size after queue.get(): ' + str(self.post_train_queue.qsize()))

                if model is None:
                    # A sign that all was done. Exit the endless loop.
                    self.post_processing=False
                    self.logger.info('All post-process tasks done. Shutting down queue.')
                    break

                self.logger.info(f'A fresh Robson({model.dp.id}) is ready for post-processing')


                # Now do all the amazing things we can do with a trained Robson.
                # In parallel.

                ## Save it to storage
                tasks.append(
                    executor.submit(
                        model.save,
                        **dict(
                            path      = self.get_config(    'TRAINED_MODELS_PATH'),
                            dvc_path  = self.get_config('DVC_TRAINED_MODELS_PATH'),
                        )
                    )
                )

                # Save train/val/test sets in DB
                tasks.append(executor.submit(model.save_train_sets))

                # Batch predict and metrics computation
                if batch_predict:
                    tasks.append(executor.submit(self.single_batch_predict,model))

            for task in concurrent.futures.as_completed(tasks):
                # Let finished tasks express themselves. This is where in-task
                # exceptions are raised.

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
            tasks.append(executor.submit(model.compute_and_save_metrics))

            # Task 2: save estimations
            tasks.append(executor.submit(model.save_batch_predict_valuations))

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
            self.logger.info(f'Using all possible Robsons in parallel')
        else:
            self.logger.info(f'Using no more than {max_workers} Robsons in parallel')

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



    def update_currents_file(self):
        if hasattr(self,'trained') and hasattr(self,'trained_in_session'):
            # Make any needed initialization
            if 'history' not in self.currents:
                self.currents['history']=[]

            if 'estimators' not in self.currents:
                self.currents['estimators']=dict()
            
            # Update file with what we have
            self.currents['history'].append(
                dict(
                    time              = datetime.datetime.now(datetime.timezone.utc),
                    user_name         = pwd.getpwuid(os.getuid())[0],
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
            self.currents['history']=self.currents['history'][-10:]
            
            for e in self.trained_in_session:
                if e in self.currents['estimators']:
                    del self.currents['estimators'][e]
                self.currents['estimators'][e]=self.trained[e].signature()
            
            target=pathlib.Path(self.get_config('PROJECT_HOME')).resolve() / 'currents.yaml'
            with open(target, 'w') as f:
                yaml.dump(
                    data=self.currents,
                    stream=f,
                    
                    # YAML styling options
                    explicit_start=True,
                    canonical=False,
                    indent=6,
                    default_flow_style=False
                )
            
            self.git_commit_currents_file()
        
        
        
    def load_currents_file(self) -> dict:
        target=pathlib.Path(self.get_config('PROJECT_HOME')).resolve() / 'currents.yaml'
        
        try:
            self.logger.debug("Trying to open {}".format(target))
            with open(target) as f:
                self.currents=yaml.safe_load(f)
        except FileNotFoundError as e:
            self.currents={}
            
        if self.currents is None:
            self.currents={}
        
        return self.currents

    

    def git_commit_currents_file(self):
        if self.get_config('COMMIT_CURRENTS', default=False, cast=bool):
            git_shell = """
                cd {path};
                git add 'currents.yaml';
                git commit -m 'chore(currents.yaml)';
                git push;
            """

            git_shell=git_shell.format(path=pathlib.Path(self.get_config('PROJECT_HOME')).resolve())
            self.logger.debug(f'Commit currents.yaml to Git with command:\n{git_shell}')
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
                    robson_module:                        str  = 'robson',
                    robson_class:                         str  = 'Robson'
        ):
        """
        Load pre-trained Robson estimators into RobsonCoach’s inventory.
        
        PKLs will be searched in S3 or filesystem path defined in
        pre_trained_path.
        
        explicit_list -- Load estimators only for this DataProviders. Load all
        if empty or None.
        
        post_process -- If True, submit loaded estimator for post-processing
        (batch predict, metrics computation etc)
        """
        robson_module = importlib.import_module(robson_module)
        Robson = getattr(robson_module, robson_class)

        robson_params_template=dict(
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
        self.trained={}

        if explicit_list is None:
            explicit_list=list(self.dp_factory.produce())

        # Find all pre-reqs of desired models and add them to the list
        to_load=set()
        for dp in explicit_list:
            if isinstance(dp, str):
                to_load=to_load | {dp}    | self.dp_factory.get_pre_req(dp)
            else:
                to_load=to_load | {dp.id} | self.dp_factory.get_pre_req(dp)

        self.logger.debug(f'Going to load pre-trained: {to_load}')

        # Parallel load of all desired pre-trained Robson objects
        with concurrent.futures.ThreadPoolExecutor(thread_name_prefix='team_load') as executor:
            tasks=[]

            for dp in to_load:
                robson_params=copy.deepcopy(robson_params_template)
                robson_params['coach']=self
                robson_params['dp']=self.dp_factory.dps[dp]() if isinstance(dp, str) else dp

                if (
                            'estimators' in self.currents and 
                            dp in self.currents['estimators']
                ):
                    robson_params['pre_trained_train_id']=self.currents['estimators'][dp]['train_id']
                    robson_params['train_or_session_id_match_list']=None

                self.logger.info(f'Trigger background loading of ‘{dp}’')
                tasks.append(executor.submit(Robson, **robson_params))

            for task in concurrent.futures.as_completed(tasks):
                # Does nothing if thread succeeded. Raises the task's exception otherwise.
                bob=task.result()
                self.declare_trained(bob, post_process=post_process)
                self.logger.info(f'Loaded {bob}')

        # Now bind pre-req models with the ones already in the team.
        for model in self.trained:
            self.trained[model].load_pre_req_model()



    ######################################################################################
    ##
    ## Metrics report
    ##
    ######################################################################################


    def report(self, train_ids: list=None, dataprovider_id: str=None, on: str='model',
                    start: str=None, reference_train_id: str=None) -> pd.DataFrame:

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


    def get_db_connection(self, nickname='robson'):
        if nickname not in self.databases:
            raise NotImplementedError(f'RobsonCoach knows nothing about a datasource with nickname «{nickname}»')
        
        import pyathena.pandas.cursor

        engine_config_sets=dict(
            # Documentation for all these SQLAlchemy pool control
            # parameters: https://docs.sqlalchemy.org/en/14/core/engines.html#engine-creation-api

            DEFAULT=dict(
                # QueuePool config for a real database
                poolclass         = sqlalchemy.pool.QueuePool,

                # 5 is the default. Reinforce default, which is good
                pool_size         = 5,

                # Default here was 10, which might be low sometimes, so
                # increase to some big number in order to neve let the
                # QueuePool be a bottleneck.
                max_overflow      = 50,
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
            athena=dict(
                connect_args=dict(
                    cursor_class=pyathena.pandas.cursor.PandasCursor
                )
            )
        )
        
        if 'conn' not in self.databases[nickname]:
            # Connection to this database not open yet. Just do it.
            
            url = self.get_config(self.databases[nickname]['env'])
        
            engine_config=engine_config_sets['DEFAULT'].copy()
            
            for dbtype in engine_config_sets.keys():
                # Extract from engine_config_sets configuration specific for each DB type
                if dbtype in url:
                    engine_config.update(engine_config_sets[dbtype])
            
            # Databricks needs special URL handling
            # URLs are like "databricks+connector://host.com/default?http_path=/sql/..."
            if 'databricks' in url:
                tokenized=urllib.parse.urlparse(url)
                
                # Extract connection args as "?http_path=..." into a dict
                # Returns {'http_path': ['value']}
                conn_args=urllib.parse.parse_qs(tokenized.query)
                
                # Unwrap the value from the list
                # Return  {'http_path': 'value'}
                engine_config.update(
                    dict(
                        connect_args={k:conn_args[k][0] for k in conn_args}
                    )
                )
                
                # Reconstruct the URL without the connection args
                tokenized=tokenized._replace(query=None)
                url=urllib.parse.urlunparse(tokenized)
    
            self.databases[nickname]['conn']=sqlalchemy.create_engine(
                url = url,
                **engine_config
            )
            self.logger.debug(f"Data source «{nickname}» is {self.databases[nickname]['conn']}")
        
        return self.databases[nickname]['conn']



    def init_db(self):
        if self.get_config('ROBSON_DB_URL'):
            # This is just to raise an exception if not set.
            # Can't do coach business without a DB.
            pass


        self.logger.info('Going to create tables on Robson DB')

        self.robson_db_metadata = sqlalchemy.MetaData(bind=self.get_db_connection('robson'))

        self.robson_db_table_prefix=self.get_config('ROBSON_DB_TABLE_PREFIX')

        self.tables['training'] = Table(
            self.robson_db_table_prefix + 'training',
            self.robson_db_metadata,
            Column(
                'train_session_id', String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'train_id', String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'dataprovider_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            Column('user_name', String),
            Column('host_name', String),
            Column('git_branch', String),
            Column('git_commit', String),
            Column('github_actor', String),      # GITHUB_ACTOR
            Column('github_workflow', String),   # GITHUB_WORKFLOW
            Column('github_run_id', String),     # GITHUB_RUN_ID
            Column('github_run_number', String), # GITHUB_RUN_NUMBER
            Column('x_features', String),
            Column('dataprovider_ver', String),

            UniqueConstraint(
                'train_session_id',
                'train_id',
                'dataprovider_id',
                name='train'
            )
        )


        self.tables['training_status'] = Table(
            self.robson_db_table_prefix + 'training_status',
            self.robson_db_metadata,
            Column(
                'time',
                Integer,
                primary_key=True,
                nullable=False
            ),
            Column(
                'dataprovider_id', sqlalchemy.String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'train_session_id', String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'train_id', String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'status',
                String,
                primary_key=True
            ),
            Column('estimator', String),
            Column('hyperparam', String),

            ForeignKeyConstraint(
                [
                    'dataprovider_id',
                    'train_session_id',
                    'train_id'
                ],
                [
                    self.robson_db_table_prefix + 'training.dataprovider_id',
                    self.robson_db_table_prefix + 'training.train_session_id',
                    self.robson_db_table_prefix + 'training.train_id',
                ]
            ),

            Index(self.robson_db_table_prefix + 'training_status_' + 'by_time', 'time'),
            Index(self.robson_db_table_prefix + 'training_status_' + 'by_dataprovider_id', 'dataprovider_id'),
            Index(self.robson_db_table_prefix + 'training_status_' + 'by_train_session_id', 'train_session_id'),
            Index(self.robson_db_table_prefix + 'training_status_' + 'by_train_id', 'train_id'),
            Index(self.robson_db_table_prefix + 'training_status_' + 'by_status', 'status'),
        )


        self.tables['sets'] = Table(
            self.robson_db_table_prefix + 'sets',
            self.robson_db_metadata,
            Column(
                'dataprovider_id', String,
#                 primary_key=True,
                nullable=False
            ),
            Column(
                'train_session_id', String,
#                 primary_key=True,
                nullable=False
            ),
            Column(
                'train_id', String,
#                 primary_key=True,
                nullable=False
            ),
            Column(
                'set', String,
                nullable=False
            ),
            Column(
                'unit_id', String,
#                 primary_key=True,
                nullable=False
            ),
            Column('target', Float,),

            ForeignKeyConstraint(
                [
                    'dataprovider_id',
                    'train_session_id',
                    'train_id'
                ],
                [
                    self.robson_db_table_prefix + 'training.dataprovider_id',
                    self.robson_db_table_prefix + 'training.train_session_id',
                    self.robson_db_table_prefix + 'training.train_id',
                ]
            ),

            Index(self.robson_db_table_prefix + 'sets_' + 'by_dataprovider_id', 'dataprovider_id'),
            Index(self.robson_db_table_prefix + 'sets_' + 'by_train_session_id', 'train_session_id'),
            Index(self.robson_db_table_prefix + 'sets_' + 'by_train_id', 'train_id'),
        )


        self.tables['metrics_model'] = Table(
            self.robson_db_table_prefix + 'metrics_model',
            self.robson_db_metadata,
            Column(
                'time',
                Integer,
                primary_key=True,
                nullable=False
            ),
            Column(
                'dataprovider_id', String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'train_session_id', String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'train_id', String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'set', String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'name', String,
                primary_key=True,
                nullable=False
            ),
            Column('value_number', Float),
            Column('value_text', String),

            ForeignKeyConstraint(
                [
                    'dataprovider_id',
                    'train_session_id',
                    'train_id'
                ],
                [
                    self.robson_db_table_prefix + 'training.dataprovider_id',
                    self.robson_db_table_prefix + 'training.train_session_id',
                    self.robson_db_table_prefix + 'training.train_id',
                ]
            ),

            Index(self.robson_db_table_prefix + 'metrics_model_' + 'by_time', 'time'),
            Index(self.robson_db_table_prefix + 'metrics_model_' + 'by_dataprovider_id', 'dataprovider_id'),
            Index(self.robson_db_table_prefix + 'metrics_model_' + 'by_train_session_id', 'train_session_id'),
            Index(self.robson_db_table_prefix + 'metrics_model_' + 'by_train_id', 'train_id'),
        )


        self.tables['metrics_valuation'] = Table(
            self.robson_db_table_prefix + 'metrics_valuation',
            self.robson_db_metadata,
            Column(
                'time',
                Integer,
                primary_key=True,
                nullable=False
            ),
            Column(
                'train_session_id', String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'train_id', String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'dataprovider_id', String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'unit_id', String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'name', String,
                primary_key=True,
                nullable=False
            ),
            Column('value_number', Float),
            Column('value_text', String),

            ForeignKeyConstraint(
                [
                    'dataprovider_id',
                    'train_session_id',
                    'train_id'
                ],
                [
                    self.robson_db_table_prefix + 'training.dataprovider_id',
                    self.robson_db_table_prefix + 'training.train_session_id',
                    self.robson_db_table_prefix + 'training.train_id',
                ]
            ),

            Index(self.robson_db_table_prefix + 'metrics_valuation_' + 'by_time', 'time'),
            Index(self.robson_db_table_prefix + 'metrics_valuation_' + 'by_dataprovider_id', 'dataprovider_id'),
            Index(self.robson_db_table_prefix + 'metrics_valuation_' + 'by_train_session_id', 'train_session_id'),
            Index(self.robson_db_table_prefix + 'metrics_valuation_' + 'by_train_id', 'train_id'),
        )


        self.tables['valuations'] = Table(
            self.robson_db_table_prefix + 'valuations',
            self.robson_db_metadata,
            Column(
                'time',
                Integer,
                primary_key=True,
                nullable=False
            ),
            Column(
                'dataprovider_id', String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'train_session_id', String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'train_id', String,
                primary_key=True,
                nullable=False
            ),
            Column(
                'unit_id', String,
                primary_key=True,
                nullable=False
            ),
            Column('sigma', Float),
            Column('p_05', Float),
            Column('p_10', Float),
            Column('p_20', Float),
            Column('p_30', Float),
            Column('p_40', Float),
            Column('p_50', Float),
            Column('p_60', Float),
            Column('p_70', Float),
            Column('p_80', Float),
            Column('p_90', Float),
            Column('p_95', Float),

            ForeignKeyConstraint(
                [
                    'dataprovider_id',
                    'train_session_id',
                    'train_id'
                ],
                [
                    self.robson_db_table_prefix + 'training.dataprovider_id',
                    self.robson_db_table_prefix + 'training.train_session_id',
                    self.robson_db_table_prefix + 'training.train_id',
                ]
            ),

            Index(self.robson_db_table_prefix + 'valuations_' + 'by_time', 'time'),
            Index(self.robson_db_table_prefix + 'valuations_' + 'by_dataprovider_id', 'dataprovider_id'),
            Index(self.robson_db_table_prefix + 'valuations_' + 'by_train_session_id', 'train_session_id'),
            Index(self.robson_db_table_prefix + 'valuations_' + 'by_train_id', 'train_id'),
        )

        self.robson_db_metadata.create_all()