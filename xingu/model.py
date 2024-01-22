import os
import os.path
import textwrap
import inspect
import socket
import urllib
import lzma
import logging
import datetime
import re
import glob
import pathlib
import hashlib
import json
import concurrent.futures
import pickle

import pandas

from . import DataProvider
from . import Coach
from . import ConfigManager






class Model(object):

    def __init__(self,
                    dp:                            DataProvider,
                    coach:                         Coach = None,

                    # Control pre-trained model loading:
                    ## Load a pre-trained model
                    trained:                              bool = False,

                    ## Load only the Estimator object
                    estimator_only:                       bool = False,
                    pre_trained_path:                     str  = None,
                    pre_trained_train_session_id:         str  = '*',
                    pre_trained_train_id:                 str  = '*',
                    pre_trained_as_of:                    str  = '*',
                    train_or_session_id_match_list:       list = None,
                    hyperopt_strategy:                    str  = 'last',

                    delayed_prereq_binding:               bool = False
                ):
        """
        Parameters:

        - dp: A DataProvider object or an ID (str) of a DataProvider to load a
        pre-trained Model for this DataProvider.

        - coach: A Coach object. Needed if Model will be trained. Needed if Model
        will do database operations. Not needed if loading a pre-trained Model.

        - estimator_class: A Estimator-derived class name to be trained by Model.
        Not required if loading a pre-trained Model.

        - trained: Causes Model to load a pre-trained object or not. The default (False)
        will give an untrained Model object.

        - pre_trained_path: Local or S3 path to search for pre-trained objects and/or to
        save them after training.

        - estimator_only: False loads and returns an entire Model object
        as it was pickled, complete with methods and pickled DataProvider
        object. True, loads only the estimator object and some IDs,
        DataProvider object and methods will not come from pickle. Use
        with caution because loaded estimator might be incompatible with
        current Model and DataProvider classes and their attributes.

        - hyperopt_strategy: How to handle estimator hyper-parameters optimization:
            - 'self' - cause it to optimize (a lengthy process).
            - 'last' (default)  - cause it to search DB the latest compatible
            hyper-parameters set, matching the DataProvider.
            - train_id string - cause it to get from DB a specific train_id
            hyper-parameters set.
            - None - use Estimator object internal defaults

        - delayed_prereq_binding: If DP specifies pre-req models, do not load them too.
        Instead, a later call to self.load_pre_req_model() will be necessary.
        """

        # Setup logging
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        # Setup coach
        self.coach=coach

        # What type of context I'm in? Might be None, "train" or "batch_predict"
        self.context=None

        self.hyperopt_strategy=hyperopt_strategy
        if hyperopt_strategy == 'last':
            self.hyperopt_strategy=self.get_config('HYPEROPT_STRATEGY', default=hyperopt_strategy)
        self.hyperparam=None

        if trained:
            # Requested a pre-trained Model. Lets try to load it

            if isinstance(dp, str):
                dp_id=dp
            elif issubclass(dp.__class__, DataProvider):
                dp_id=dp.id

            bob=Model.load(
                dp_id                           = dp_id,
                path                            = pre_trained_path,
                train_session_id                = pre_trained_train_session_id,
                train_id                        = pre_trained_train_id,
                train_or_session_id_match_list  = train_or_session_id_match_list,
                as_of                           = pre_trained_as_of
            )

            # Transplant loaded object attributes into current object
            if estimator_only:
                # Use the DP in RAM, not from pickle
                if issubclass(dp.__class__, DataProvider):
                    self.dp        = dp
                else:
                    raise Exception(
                        "This type of Model loading requires a DataProvider "
                        "object, not just a string. Use "
                        "DataProviderFactory.get('model_name') to get it."
                    )
            else:
                self.dp            = bob.dp

            # Copy a managed list of attributes from loaded object
            attributes=[
                # Train info
                'train_id',                 'train_session_id',    'estimator',
                'trained',                  'hyperopt_strategy',   'hyperparam',
                'train_queries_signatures',

                # OS environment info
                'user_name',      'host_name',

                # Code info
                'git_branch',     'git_commit',

                # GitHub Actions info
                'github_actor',   'github_workflow',
                'github_run_id',  'github_run_number'
            ]

            for a in attributes:
                if hasattr(bob,a):
                    setattr(self,a,getattr(bob,a))

            del bob
        else:
            # Requested an uninitalized/untrained object
            self.dp                       = dp
            self.trained                  = trained
            self.train_id                 = None
            self.train_session_id         = None
            self.train_queries_signatures = dict()

            # Store more contextual and informative metadata from the Coach and environment
            self.user_name         = self.get_config('USERNAME',os.environ.get('USER', os.environ.get('USERNAME')))
            self.host_name         = self.get_config('HOSTNAME',socket.gethostname())
            self.git_branch        = self.coach.git_repo.head.name if self.coach.git_repo else None
            self.git_commit        = self.coach.git_repo.head.target.hex if self.coach.git_repo else None
            self.github_actor      = self.get_config('GITHUB_ACTOR', None)
            self.github_workflow   = self.get_config('GITHUB_WORKFLOW', None)
            self.github_run_id     = self.get_config('GITHUB_RUN_ID', None)
            self.github_run_number = self.get_config('GITHUB_RUN_NUMBER', None)

            self.log(message=str(self.dp.get_estimator_class()))

            if self.dp.hollow:
                self.estimator     = None
            else:
                self.estimator     = self.dp.get_estimator_class()(
                    **self.dp.get_estimator_class_params(),
                    params      = self.dp.get_estimator_params(),
                    hyperparams = self.dp.get_estimator_hyperparams(),
                )

        if delayed_prereq_binding is False:
            self.load_pre_req_model()



    ######################################################################################
    ##
    ## Train a Model object, including hyperparam optimization
    ##
    ######################################################################################



    def hyperparam_optimize(self):
        """
        Decide wether to compute or use previously computed hyperparameters.

        This is controlled by self.hyperopt_strategy, which can be:

        - self -- compute hyperparameters optimization
        - last -- search Model DB for last computed hyperparams for this
            DataProvider
        - dp -- use hyperparams as returned by self.dp.get_estimator_hyperparameters()
        - a train name -- load from Xingu DB the previously computed
            hyperparams used on a specific train_id
        """

        self.context='train_hyperopt'

        # Declare we have no clue of hyper-params
        self.algo_params=None
        dbresult=None

        # Start handling hyperparameters
        self.log_train_status('train_hyperhandle_start')

        if self.hyperopt_strategy == 'self':
            # Lengthy estimator hyper-parameter optimization
            self.log_train_status('train_hyperopt_start')

            self.log(message='Hyperparameter optimization...')

            # This single call takes a long time to process
            h=self.estimator.hyperparam_optimize(
                model          = self
                # datasets     = self.sets,
                # features     = self.dp.get_estimator_features_list(),
                # target       = self.dp.get_target(),
                # search_space = self.dp.get_estimator_optimization_search_space()
            )

            self.algo_params = dict(
                source_train_id = self.hyperopt_strategy,
                params          = self.dp.get_estimator_params() or self.estimator.params,
                hyperparams     = h,
            )

            self.log_train_status('train_hyperopt_end')


        elif self.hyperopt_strategy == 'dp':
            # DataProvider defines algorithm parameters and hyperparameters

            self.algo_params = dict(
                source_train_id = self.hyperopt_strategy,
                params          = self.dp.get_estimator_params() or self.estimator.params,
                hyperparams     = self.dp.get_estimator_hyperparams() or self.estimator.hyperparams,
            )


        elif self.hyperopt_strategy is not None:
            # Get from DB last computed hyper-parameters for this DP or a specific train_id.

            import sqlalchemy

            # Table shortcuts for code readability
            attributes = self.coach.tables['training_attributes']
            steps      = self.coach.tables['training_steps']

            query=(
                sqlalchemy.select(
                    [
                        steps.c.time,
                        attributes.c.dataprovider_id,
                        attributes.c.train_session_id,
                        attributes.c.train_id,
                        attributes.c.attribute,
                        attributes.c.value
                    ]
                )
                .join(
                    steps,
                    sqlalchemy.and_(
                        attributes.c.dataprovider_id  == steps.c.dataprovider_id,
                        attributes.c.train_session_id == steps.c.train_session_id,
                        attributes.c.train_id         == steps.c.train_id,
                    )
                )
                .where(
                    sqlalchemy.and_(
                        attributes.c.dataprovider_id==self.dp.id,
                        steps.c.status=='train_hyperhandle_end',
                        attributes.c.attribute=='hyperparam',
                    )
                )
                .order_by(steps.c.time.desc())
                .limit(1)
            )

            if self.hyperopt_strategy != 'last':
                # Include additional condition
                query=query.where(
                    sqlalchemy.or_(
                        attributes.c.train_id         == self.hyperopt_strategy,
                        attributes.c.train_session_id == self.hyperopt_strategy,
                    )
                )

            self.log(
                level=logging.DEBUG,
                message=(
                    'Searching DB for optimized hyperparameters as:\n' +
                    query.compile(compile_kwargs={'literal_binds': True}).string
                )
            )

            dbresult = query.execute().first()

            if dbresult is not None:
                # Import the JSON text into a dict
                self.algo_params=json.loads(dbresult.value)

                if self.algo_params['source_train_id']=='self':
                    # Give credit for the source of hyper-params
                    self.algo_params['source_train_id']=dbresult.train_id

        # At this point we have hyperparam self-computed, or retrieved from DB, or None
        if self.algo_params is None:
            # If still None, get defaults from estimator
            self.algo_params=dict(
                source_train_id = 'default',
                params          = self.estimator.params,
                hyperparams     = self.estimator.hyperparams,
            )
        else:
            # If we have something, set it on xingu.Estimator as an ordered
            # combination of:
            # - DataProvider default paramters (including operational ones)
            # - Searched or found parameters
            # The last ones overwrite the first ones

            self.estimator.params = {
                i[0]:i[1]
                for i in
                list(self.dp.get_estimator_params().items()) +
                list(self.algo_params['params'].items())
            }

            self.estimator.hyperparams = {
                i[0]:i[1]
                for i in
                list(self.dp.get_estimator_hyperparams().items()) +
                list(self.algo_params['hyperparams'].items())
            }

        self.log_train_status('train_hyperhandle_end')



    def fit(self):
        """
        Train Model configured by its DataProvider. Steps:

        - Get DP SQL queries
        - Use queries to get data from DBs
        - Pass data to DP in order to let it clean and integrate data into one DataFrame
        - Ask DP to do feature engineering
        - Ask DP to split data into train/test/valuation
        - Train a Estimator with splited data
        - Get some metrics
        - Save trained model pickle
        """
        import randomname

        self.train_id          = randomname.get_name()
        if self.coach:
            self.train_session_id  = self.coach.train_session_id

        self.context='train_dataprep'
        self.log_train_status('train_dataprep_start')

        # Get DP SQL queries
        self.log(message='Asking DataProvider for training queries')
        sources=self.dp.get_dataset_sources_for_train(model=self)

        # Transform queries into real data
        self.log(message='Retrieving database data as defined by DataProvider')
        train_raw_datasets=self.data_sources_to_data(sources)

        # Data integration and clenup
        self.log(message='DataProvider will clean data')
        train_data=self.dp.clean_data_for_train(train_raw_datasets)

        # Feature engineering
        self.log(message='DataProvider will do some feature engineering')
        train_data=self.dp.feature_engineering_for_train(train_data)
        self.log(
            level=logging.DEBUG,
            message='Feature engineering returned columns: \n' + pandas.Series(train_data.columns).to_markdown()
        )

        # Give it a chance to post-process data before training
        train_data=self.dp.last_pre_process_for_train(train_data)

        # Train/test split
        self.log(message='DataProvider will split data')
        self.sets=self.dp.data_split_for_train(train_data)

        if self.sets is not None:
            self.save_sets_cache()

            report=[]
            for k in self.sets.keys():
                report.append(
                    "{theset}=({l} lines, {b} bytes)".format(
                        theset=k,
                        b=self.sets[k].memory_usage(deep=True).sum(),
                        l=self.sets[k].shape[0]
                    )
                )
            report='; '.join(report)

            self.log(message=f'Sizes of train sets: {report}',level=logging.DEBUG)

        self.log_train_status('train_dataprep_end')

        if not self.dp.hollow:
            self.hyperparam_optimize()

            self.dp.post_process_after_hyperparam_optimize(self)

            self.log(
                message=f"Hyperparameters from {self.algo_params['source_train_id']}:\n" + pandas.Series(self.algo_params['hyperparams']).to_markdown(),
                level=logging.DEBUG
            )

            self.context='train_fit'
            self.log_train_status('train_fit_start')

            self.log(message='Training an Estimator')

            self.estimator.fit(
                datasets  = self.sets,
                features  = self.dp.get_estimator_features_list(),
                target    = self.dp.get_target(),
                model     = self
            )

            self.log_train_status('train_fit_end')

        self.trained=datetime.datetime.now(datetime.timezone.utc)
        # A chance for the DataProvider to manipulate the Model object or use
        # it to compute something before any PKL saving or metrics computation.
        self.dp.post_process_after_train(self)




    ######################################################################################
    ##
    ## Batch Predict
    ##
    ######################################################################################



    def batch_predict(self):
        """
        Batch predict for a dataset based on information provided by its DataProvider. Steps:

        - Get DP SQL queries
        - Use queries to get data from DBs
        - Pass data to DP in order to let it clean and integrate data into one DataFrame
        - Ask DP to do feature engineering
        - Predict
        - Save predicted data to DB
        """

        self.context='batch_predict'

        # Get DP SQL queries
        self.log('Asking DataProvider for batch predict queries')
        sources=self.dp.get_dataset_sources_for_batch_predict(model=self)
        if sources is None:
            self.log('DataProvider provides no data to batch predict')
            return None

        # Execute queries in order to get real data
        self.log('Retrieving database data as defined by DataProvider')
        raw_datasets=self.data_sources_to_data(sources)

        # Data integration and clenup
        self.log('DataProvider will clean data')
        self.batch_predict_data=self.dp.clean_data_for_batch_predict(raw_datasets)

        # Feature engineering
        self.log('DataProvider will do some feature engineering')
        self.batch_predict_data=self.dp.feature_engineering_for_batch_predict(self.batch_predict_data)
        self.log(
            level=logging.DEBUG,
            message=f'Feature engineering returned columns: {list(self.batch_predict_data.columns)}'
        )

        # Give it a chance to post-process data before batch predict
        self.batch_predict_data=self.dp.last_pre_process_for_batch_predict(self.batch_predict_data)

        # Trigger estimators
        if self.estimator.is_classifier():
            self.batch_predict_estimations=self.predict_proba(self.batch_predict_data)
        else:
            self.batch_predict_estimations=self.predict(self.batch_predict_data)

        # Record time of estimation
        self.batch_predict_time = datetime.datetime.now(datetime.timezone.utc)

        self.context=None



    ######################################################################################
    ##
    ## Predict methods
    ##
    ######################################################################################


    def predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        return self.generic_predict(data)



    def predict_proba(self, data: pandas.DataFrame, class_index: int=None) -> pandas.DataFrame:
        return self.generic_predict(data,method='predict_proba',class_index=class_index)



    def generic_predict(self, data: pandas.DataFrame, method='predict', class_index: int=None) -> pandas.DataFrame:
        """
        Calls estimator.predict() or predict_proba() (method='predict_proba').
        Let DataProvider pre-process and post-process data respectively before and
        after predict method.

        Hollow models (no estimator) use this sequence of calls to simulate predicts.

        Return Y_pred with estimations.
        """
        methods = dict(
                 predict = None if self.dp.hollow else getattr(self.estimator, method),
             pre_predict = getattr(self.dp, f'pre_process_for_{method}'),
            post_predict = getattr(self.dp, f'post_process_after_{method}'),
        )

        # Do whatever pre-processing the DataProvider wants to do
        prepared=methods['pre_predict'](X=data.copy(), model=self)

        # Call predict method only if we are not hollow
        params=dict()
        if method=='predict_proba':
            params['class_index']=class_index
        Y_pred=methods['predict'](
            prepared[self.dp.get_estimator_features_list()],
            **params
        ) if methods['predict'] else None


        # Do whatever post-processing the DP wants to do.
        # Hollow Models (no estimator) take this chance to actually compute
        # Y_pred in here.
        Y_pred=(
            methods['post_predict'](
                X         = prepared,
                Y_pred    = Y_pred,
                model     = self
            )

            # Reorder to match input
            .reindex(prepared.index)
        )

        return Y_pred



    ######################################################################################
    ##
    ## Post-Fit and post-Batch Predict procedures
    ##
    ######################################################################################



    def trainsets_save(self):
        def ddebug(input_df, message):
            self.log(message,level=logging.DEBUG)
            return input_df

        if self.sets is not None:
            self.context='train_savesets'
            self.log_train_status('train_savesets_start')

            with self.coach.get_db_connection('xingu').connect() as db:
                # Record sets in DB
                for part in self.sets.keys():
                    self.log(f'Saving {part} dataset to DB')

                    target=self.dp.get_target()
                    if target in self.sets[part].columns:
                        (
                            # Start with just the index and target column
                            self.sets[part][[target]]

                            # Standardize target column name
                            .rename(columns={target: 'target'})

                            # Tag table with context
                            .assign(
                                set                = part,
                                dataprovider_id    = self.dp.id,
                                train_id           = self.train_id,
                                train_session_id   = self.train_session_id,
                            )

                            # Standardize index
                            .reset_index(names='index')

                            # Force data types, some DBs (cough Athena cough) will complain otherwise
                            .assign(
                                index=lambda table: table['index'].astype(str),
                                target=lambda table: table.target.astype(float),
                            )

                            # Write debug message to console
                            .pipe(lambda table: ddebug(
                                table,
                                f'{part}: {table.shape[0]}×{table.shape[1]}'
                            ))

                            # Commit to DB
                            .to_sql(
                                self.coach.tables['sets'].name,
                                if_exists='append',
                                index=False,
                                con=db
                            )
                        )

            self.log_train_status('train_savesets_end')



    def trainsets_predict(self):
        if not hasattr(self, 'sets'):
            return

        if not hasattr(self, 'sets_estimations'):
            # Create sets_estimations only if still doesn´t exist
            # It might have been created in the train and validation process
            # by the Estimator.
            self.sets_estimations = dict()

        for part in self.sets.keys():
            if part not in self.sets_estimations:
                # If already not prodicted (e.g. 'validation')
                self.log(f'Predicting «{part}» part of the training dataset for metrics purposes')

                if self.estimator is not None and self.estimator.is_classifier():
                    self.sets_estimations[part] = self.predict_proba(self.sets[part])
                else:
                    self.sets_estimations[part] = self.predict(self.sets[part])
            else:
                self.log(f'Prediction of «{part}» was already made by some previous task in the Xingu pipeline, probably internals of your xingu.Estimator.')

        self.dp.pre_process_for_trainsets_metrics(self)
        self.compute_and_save_metrics(channel='trainsets')
        self.dp.post_process_after_trainsets_metrics(self)


    def save_batch_predict_estimations(self):
        if hasattr(self,'batch_predict_estimations'):
            self.log('Save predicted estimations to DB::estimations table')

            if self.estimator.is_classifier():
                # predict_proba() returns multiple columns, one for each
                # classification class. But current DB design only
                # supports 1 Y_pred value. So drop all Y_pred columns
                # except DP.proba_class_index

                estimation_col=f'estimation_class_{self.dp.proba_class_index}'
            else:
                estimation_col='estimation'

            with self.coach.get_db_connection().connect() as db:
                (
                    self.batch_predict_estimations

                    # Work on a shallow copy
                    .copy(deep=False)

                    .assign(
                        # Decide about which estimation column to use
                        # (in case of a classifier)
                        estimation       = lambda table: table[estimation_col],
                    )

                    # Standardize index
                    .reset_index(names='index')

                    # Only want these columns
                    [['index', 'estimation']]

                    .assign(
                        # Tag rows with context
                        time             = round(self.batch_predict_time.timestamp()),
                        train_id         = self.train_id,
                        train_session_id = self.train_session_id,
                        dataprovider_id  = self.dp.id,
                    )

                    # Commit to DB
                    .to_sql(
                        con          = db,
                        name         = self.coach.tables['estimations'].name,
                        if_exists    = 'append',
                        index        = False,
                    )
                )



    ######################################################################################
    ##
    ## Metrics handling
    ##
    ######################################################################################


    # There are 2 types of Error Metrics:
    #
    # • Model metrics, as classical RMSE, MAPE, but also OKR 15%
    # • Per-valuation metrics, as Farol and Score
    #
    # Metrics are computed from various types of data:
    #
    # • Training sets with Y (as train, test, val) against estimated Ŷ, as classical metrics
    # • Arbitrary data with Y against estimated Ŷ, for metric as OKR 15%
    # • Pure uni-Model simple estimations, as Score
    # • Estimations across multiple models, as Farol
    #
    # These are the ones that we do and know today and Model tries to provide a framework
    # to handle all these kinds of metrics and, hopefully, new funky types that may appear
    # in the future. But it is hard to accomodate unknown future freestyle requirements.
    #
    # Model will scan itself and its DataProvider for methods with following signature:
    #
    # • compute_trainsets_model_metrics_{NAME}(self, XY_set, Y_pred)
    #   Return: dict(metric_name: value, metric_name: value, ...)
    #   Example: MAPE, quantile deviation, interval (Model)
    #
    # • compute_global_model_metrics_{NAME}(self, XY, Y_pred)
    #   Return: dict(metric_name: value, metric_name: value, ...)
    #   Example: OKR 15% (DP), Farol (DP)
    #
    # • compute_estimation_metrics_{NAME}(self, XY, Y_pred)
    #   Return: DataFrame(Index(unit_id), name, value_number, value_text)
    #   Example: Score (Model), Farol (DP)
    #
    # These methods can compute and return multiple metrics each. Just implement and it
    # will be called. Implement it in Model class if it makes sense to all models, such
    # as the classic metrics. Implement specific metrics into its DataProvider. Implement
    # it also in a parent abstract DataProvider using attributes from concrete derived
    # classes to attain code elegance.
    #
    # Avoid {NAME} clashes between Model and DataProvider implementations, bless them
    # with descriptive and appropriate name.
    #
    # Valuation metrics methods (Score, Farol) can be used in single estimation operations
    # too, not only in the train and batch predict pipelines.
    #
    # Model metrics methods have a better fit in train and batch predict sessions, but,
    # if you put the right data in place, they can be used also outside these contexts.
    #
    # These are the drivers of the design.
    #
    # METHODS AND THEIR CALL STACK
    #
    # self.compute_and_save_metrics() calls:
    #     self.save_model_metrics() calls:
    #             self.compute_model_metrics() calls:
    #                 self.compute_trainsets_model_metrics() calls (Model.predict() is called here):
    #                     All self.compute_trainsets_model_metrics_{NAME}()
    #                     All self.dp.compute_trainsets_model_metrics_{NAME}()
    #                 self.compute_global_model_metrics() calls:
    #                     All self.compute_global_model_metrics_{NAME}()
    #                     All self.dp.compute_global_model_metrics_{NAME}()
    #     self.save_estimation_metrics() calls:
    #             self.compute_estimation_metrics() calls:
    #                 All self.compute_estimation_metrics_{NAME}()
    #                 All self.dp.compute_estimation_metrics_{NAME}()




    def get_metrics_computers(self, type: str='estimation') -> list:
        """
        Return a list of member functions that can compute metrics.

        Current known types are:

        • trainsets_model
        • batch_model
        • global_model
        • estimation
        """

        methods=[]
        for method in inspect.getmembers(self, predicate=inspect.ismethod):
            if 'compute_' + type + '_metrics_' in method[0]:
                methods.append(method[1])

        return methods



    def get_plot_renderers(self, type: str='global_model') -> list:
        """
        Return a list of member functions that can render plots.

        Current known types are:

        • trainsets_model
        • batch_model
        • global_model
        """

        methods=[]
        for method in inspect.getmembers(self, predicate=inspect.ismethod):
            if 'render_' + type + '_plots_' in method[0]:
                methods.append(method[1])

        return methods



    def compute_trainsets_model_metrics(self) -> dict:
        """
        Calls all compute_trainsets_model_metrics_{NAME}() from Model and
        DataProvider to compute metrics for self.sets DataFrames.
        """

        all_metrics={}

        if hasattr(self, 'sets'):
            methods = dict(
                model   = self.get_metrics_computers(type='trainsets_model'),
                dp      = self.dp.get_metrics_computers(type='trainsets_model')
            )

            # Collect metrics for trained model, making it predict values for
            # known and unknown data
            for part in self.sets.keys():
                self.log(f'Compute error metrics for {part} part of the training dataset')

                all_metrics[part]={}

                for domain in methods.keys():
                    params=dict(
                        XY      = self.sets[part],
                        Y_pred  = self.sets_estimations[part]
                    )

                    # Add model object as parameter if this is DP-provided method
                    if domain == 'dp':
                        params['model']=self

                    for train_metrics_computer in methods[domain]:
                        all_metrics[part].update(
                            train_metrics_computer(**params)
                        )

        return all_metrics



    def compute_batch_model_metrics(self) -> dict:
        """
        Calls all compute_batch_model_metrics_{NAME}() from Model and DataProvider.
        """

        methods = dict(
            model   = self.get_metrics_computers(type='batch_model'),
            dp      = self.dp.get_metrics_computers(type='batch_model')
        )

        metrics={}
        for domain in methods.keys():
            # Add model object as parameter if this is DP-provided method
            params=dict(
                XY       = self.batch_predict_data,
                Y_pred   = self.batch_predict_estimations
            )
            if domain == 'dp':
                params['model']=self

            for batch_metrics_computer in methods[domain]:
                metrics.update(batch_metrics_computer(**params))

        return metrics



    def compute_global_model_metrics(self) -> dict:
        """
        Calls all compute_global_model_metrics_{NAME}() from Model and DataProvider.
        """

        methods = dict(
            model   = self.get_metrics_computers(type='global_model'),
            dp      = self.dp.get_metrics_computers(type='global_model')
        )

        metrics={}
        for domain in methods.keys():
            # Add model object as parameter if this is DP-provided method
            params=dict()
            if domain == 'dp':
                params['model']=self

            for global_metrics_computer in methods[domain]:
                metrics.update(global_metrics_computer(**params))

        return metrics



    def compute_model_metrics(self, channel='trainsets') -> pandas.DataFrame:
        """
        Calls all compute_trainsets_model_metrics_{NAME}() and
        compute_global_model_metrics_{NAME}() from Model and DataProvider.

        All dict to DataFrame handling happens here.
        """

        self.log('Compute model metrics.')

        model_metrics = {}

        # Get metrics for train, test and val sets
        if (channel=='trainsets' or channel=='global') and hasattr(self, 'sets') and self.sets is not None:
            # Operate only if we are into a training session, only if we
            # have actual train/test data.
            model_metrics.update(self.compute_trainsets_model_metrics())
            model_metrics['global']=self.compute_global_model_metrics()

        # Get batch predict metrics
        elif channel == 'batch' and hasattr(self, 'batch_predict_data') and hasattr(self, 'batch_predict_estimations'):
            # Operate only if we batch predict data and estimations
            model_metrics['batch'] = self.compute_batch_model_metrics()
        else:
            self.log('No batch data to compute model metrics.', level=logging.WARNING)


        # Now convert the model_metrics dict into the metrics DataFrame.
        # Separate numeric and textual values.
        metrics=None
        for part in model_metrics:
            match_real_numbers = r'((\+|-)?([0-9]+)(\.[0-9]+)?)|((\+|-)?\.?[0-9]+)'
            met = pandas.Series(model_metrics[part])
            metricsdf = (
                # Start with only the numeric metrics
                pandas.DataFrame(
                    met[met.astype(str).str.match(match_real_numbers)]
                    .rename('value_number')
                )

                # Add columns for textual metrics
                .join(
                    how='outer',
                    other=(
                        met[~met.astype(str).str.match(match_real_numbers)]
                        .rename('value_text')
                    )
                )

                # Make sure empty cells have null
                .fillna(pandas.NA)

                .assign(
                    # Convert to appropriate type
                    value_number = lambda table: pandas.to_numeric(table.value_number),
                    set = part
                )

                # Standardize index
                .reset_index(names='name')
            )

            if metrics is None:
                metrics=metricsdf
            else:
                metrics=pandas.concat([metrics,metricsdf])

        if metrics is not None and metrics.shape[0]>0:
            self.log('Model metrics: \n' + metrics.reset_index(drop=True).to_markdown(), level=logging.DEBUG)

        return metrics



    def tag_model_plot(self, ax, template=None, loc='lower left', alpha=0.4, size=10):
        """
        Put a model signature with train metadata in your plot´s
        matplotlib.Axes.

        Default text includes DataProvider ID, train IDs and time the model
        was trained. You can change the text by passing a different
        Python format()-ready text to `template` arg.

        Define signature position with `loc`, transparency with `alpha` and
        text size with `size`.
        See matplotlib.offsetbox.AnchoredText documentation for possible
        locations.

        Returns the signed and modified Axes passed in the `ax` parameter.
        """

        import matplotlib

        if template is None:
            template='\n'.join(
                [
                    "DataProvider: {dp}",
                    "Train: {complete_train_id}",
                    "Trained: {trained}"
                ]
            )

        model_signature = matplotlib.offsetbox.AnchoredText(
            prop=dict(size=size),
            frameon=True,
            loc=loc,
            s=template.format(
                dp=self.dp.id,
                complete_train_id=self.get_full_train_id(),
                trained=self.trained.strftime("%Y-%m-%d %H:%M:%S %Z"),
            ),
        )

        model_signature.patch.set_alpha(alpha)
        model_signature.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")

        ax.add_artist(model_signature)

        return ax



    def render_trainsets_model_plots(self) -> dict:
        """
        Calls all render_trainsets_model_plots_{NAME}() from Model and
        DataProvider to render plots for each of the self.sets DataFrames.
        """

        all_plots={}

        if hasattr(self, 'sets'):
            methods = dict(
                model   = self.get_plot_renderers(type='trainsets_model'),
                dp      = self.dp.get_plot_renderers(type='trainsets_model')
            )

            # Collect metrics for trained model, making it predict values for
            # known and unknown data
            for part in self.sets.keys():
                self.log(f'Render plots for {part} part of the training dataset')

                all_plots[part]={}

                for domain in methods.keys():
                    params=dict(
                        XY      = self.sets[part],
                        Y_pred  = self.sets_estimations[part]
                    )

                    # Add model object as parameter if this is DP-provided method
                    if domain == 'dp':
                        params['model']=self

                    for train_plot_renderer in methods[domain]:
                        plots=train_plot_renderer(**params)
                        if plots is not None:
                            import matplotlib
                            for p in plots:
                                matplotlib.pyplot.close(plots[p])
                            all_plots[part].update(plots)

        return all_plots if len(all_plots.keys()) else None



    def render_batch_model_plots(self) -> dict:
        """
        Calls all render_batch_model_plots_{NAME}() from Model and
        DataProvider to render plots for the batch_predict_data.
        """

        methods = dict(
            model   = self.get_plot_renderers(type='batch_model'),
            dp      = self.dp.get_plot_renderers(type='batch_model')
        )

        all_plots = dict()

        for domain in methods.keys():
            # Add model object as parameter if this is DP-provided method
            params=dict(
                XY       = self.batch_predict_data,
                Y_pred   = self.batch_predict_estimations
            )
            if domain == 'dp':
                params['model']=self

            for batch_plot_renderer in methods[domain]:
                plots=batch_plot_renderer(**params)
                if plots is not None:
                    import matplotlib
                    for p in plots:
                        matplotlib.pyplot.close(plots[p])
                    all_plots.update(plots)

        return all_plots if len(all_plots.keys()) else None



    def render_global_model_plots(self) -> dict:
        """
        Calls all render_global_model_plots_{NAME}() from Model and
        DataProvider to render plots for whatever the DataProvider wants.
        """

        methods = dict(
            model   = self.get_plot_renderers(type='global_model'),
            dp      = self.dp.get_plot_renderers(type='global_model')
        )

        all_plots = dict()

        for domain in methods.keys():
            # Add model object as parameter if this is DP-provided method
            params=dict()
            if domain == 'dp':
                params['model']=self

            for global_plot_renderer in methods[domain]:
                plots=global_plot_renderer(**params)
                if plots is not None:
                    import matplotlib
                    for p in plots:
                        matplotlib.pyplot.close(plots[p])
                    all_plots.update(plots)

        return all_plots if len(all_plots.keys()) else None



    def render_model_plots(self, channel='trainsets') -> dict:
        """
        Calls all render_trainsets_model_plots_{NAME}() and
        render_global_model_plots_{NAME}() from Model and DataProvider.
        """
        self.log('Render model plots.')

        model_plots = dict()

        # Get plots for train, test and val sets
        if (channel=='trainsets' or channel=='global') and hasattr(self, 'sets') and self.sets is not None:
            # Operate only if we are into a training session, only if we
            # have actual train/test data.
            model_plots.update(self.render_trainsets_model_plots())
            model_plots['global']=self.render_global_model_plots()
            if model_plots['global'] is None:
                del model_plots['global']

        # Get batch predict metrics
        elif channel == 'batch' and hasattr(self, 'batch_predict_data') and hasattr(self, 'batch_predict_estimations'):
            # Operate only if we batch predict data and estimations
            model_plots['batch'] = self.render_batch_model_plots()
            if model_plots['batch'] is None:
                del model_plots['batch']

        return model_plots if len(model_plots.keys()) else None



    def render_global_model_plots_ROCAUC(self):
        """
        Plot one axes with overlapped ROC curves of all sets available on
        model.sets. Include AUC number as annotation.
        """
        if not self.estimator.is_classifier():
            # ROC curves only make sense to classifiers
            return None

        import sklearn.metrics

        target_col = self.dp.get_target()
        proba = f'estimation_class_{self.dp.proba_class_index}'

        # Make 'train' the first set
        priority = 'train'
        sets = [priority] + list(set(self.sets.keys())-set([priority]))

        # Associate some colors to datasets
        colors = """
            blue
            red
            green
            brown
            black
            yellow
            cyan
        """.split()

        # This works as a join...
        colors = dict(zip(sets,colors))

        roc = None
        for s in sets:
            # Draw many ROC curves in the same axes.
            roc=sklearn.metrics.RocCurveDisplay.from_predictions(
                y_true     = self.sets[s][target_col],
                y_pred     = self.sets_estimations[s][proba],
                pos_label  = 1,
                name       = s,
                color      = colors[s],
                ax         = (roc.ax_ if roc is not None else None)
            )

        roc.ax_.set_title("ROC and Area Under Curve")

        return (
            {
                'ROC and AUC': (
                    self.tag_model_plot(
                        roc.ax_,
                        loc='upper left'
                    )
                    .get_figure()
                )
            } if roc is not None else None
        )



    def save_model_metrics(self, channel='trainsets'):
        self.log_train_status(f'model_{channel}_metrics_start')

        metrics = self.compute_model_metrics(channel)

        if metrics is not None:
            the_time=self.trained.timestamp()
            if channel=='batch':
                the_time=round(self.batch_predict_time.timestamp())

            self.log('Saving model metrics to DB')

            with self.coach.get_db_connection('xingu').connect() as db:
                (
                    metrics

                    # Tag metrics with IDs of this training session
                    .assign(
                        time              = the_time,
                        dataprovider_id   = self.dp.id,
                        train_session_id  = self.train_session_id,
                        train_id          = self.train_id
                    )

                    # Save metrics to DB
                    .to_sql(
                        name           = self.coach.tables['metrics_model'].name,
                        if_exists      = 'append',
                        index          = False,
                        con            = db
                    )
                )

        plots = self.render_model_plots(channel)

        if plots is not None and len(plots.keys())>0:
            plot_template="{dp} • {time} • {full_train_id} • {part} • {signature}.{ext}"

            formats = self.get_config('PLOTS_FORMAT', default='svg')
            path = self.get_config('PLOTS_PATH', default='.')

            # PLOTS_FORMAT might have a single format as 'svg' but also a list of formats,
            # as 'png, svg', so we´ll save plots in all formats

            for channel in plots.keys():
                self.log('Plots: ' + str(plots))
                for plot in plots[channel].keys():
                    for ext in [f.strip() for f in formats.split(',')]:
                        plots[channel][plot].savefig(
                            fname = (
                                pathlib.Path(path) /
                                plot_template.format(
                                    dp=self.dp.id,
                                    time=Model.time_fs_str(
                                        self.batch_predict_time
                                        if channel == 'batch'
                                        else self.trained
                                    ),
                                    full_train_id=self.get_full_train_id(),
                                    part=channel,
                                    signature=plot,
                                    ext=ext
                                )
                            ),
                            dpi = 300,
                            bbox_inches = 'tight',
                            # transparent = True
                        )

        self.log_train_status(f'model_{channel}_metrics_end')



    def compute_batch_estimation_metrics(self) -> pandas.DataFrame:
        if hasattr(self, 'batch_predict_data') and hasattr(self, 'batch_predict_estimations'):
            return self.compute_estimation_metrics(self.batch_predict_data,self.batch_predict_estimations)
        else:
            self.log('No data to compute estimations metrics.', level=logging.WARNING)



    def compute_batch_model_metrics_classical(self, XY: pandas.DataFrame, Y_pred: pandas.DataFrame) -> dict:
        mask = ~XY[self.dp.get_target()].isna()
        return self.compute_trainsets_model_metrics_classical(XY[mask],Y_pred[mask])



    def compute_estimation_metrics(self, XY: pandas.DataFrame, Y_pred: pandas.DataFrame) -> pandas.DataFrame:
        """
        Calls all compute_estimation_metrics_{NAME}() from Model and DataProvider.

        Each method must return a DataFrame with following columns:
        - index (has to be the index)
        - name -- name of metric, such as 'score'
        - value_number -- metric value if numeric, such as '0.97234'
        - value_text -- metric value if it is text, such as 'red'
        """

        self.log(f'Compute estimation metrics')

        metrics=None

        methods = dict(
            model   = self.get_metrics_computers(type='estimation'),
            dp      = self.dp.get_metrics_computers(type='estimation')
        )

        for domain in methods.keys():
            for estimation_metrics_computer in methods[domain]:
                params=dict(
                    XY      = XY,
                    Y_pred  = Y_pred
                )

                if domain == 'dp':
                    params.update({'model': self})

            # Call each and every metrics computer method
            for estimation_metrics_computer in methods[domain]:
                m1=estimation_metrics_computer(**params)

                if metrics is None:
                    metrics=m1
                elif m1 is not None:
                    metrics=pandas.concat([metrics,m1])

        if metrics is not None:
            # Make sure index has correct name
            metrics.index.rename(XY.index.name, inplace=True)

            # Drop rows where BOTH value_number AND value_text have NaN
            metrics.dropna(
                subset={'value_number','value_text'}.intersection(metrics.columns),
                how='all',
                inplace=True
            )
        else:
            self.log('No estimations metrics computer methods.', level=logging.WARNING)

        return metrics



    def save_estimation_metrics(self, channel='trainsets'):
        """
        Trigger computation of estimation metrics and save them on DB.
        """

        if channel=='batch_predict':
            self.log(f'Preparing for estimations metrics')

            metrics=None

            metrics=self.compute_batch_estimation_metrics()
            self.log(f'Estimation metrics computed')

            if metrics is not None:
                # Tag metrics with some context
                metrics['time']               = round(self.batch_predict_time.timestamp())
                metrics['train_id']           = self.train_id
                metrics['train_session_id']   = self.train_session_id
                metrics['dataprovider_id']    = self.dp.id

                self.log(f'Save estimations metrics to DB')

                with self.coach.get_db_connection('xingu').connect() as db:
                    metrics.reset_index(names='index').to_sql(
                        name         = self.coach.tables['metrics_estimation'].name,
                        if_exists    = 'append',
                        index        = False,
                        con          = db
                    )



    def compute_and_save_metrics(self, channel='trainsets'):
        """
        Trigger all metrics computations.
        Can act on trainsets (self.sets and self.sets_estimations) and
        on data from the batch predict service (self.batch_predict_data).
        """
        self.save_model_metrics(channel)
        self.save_estimation_metrics(channel)



    def compute_trainsets_model_metrics_classical(self, XY: pandas.DataFrame, Y_pred: pandas.DataFrame) -> dict:
        """
        Compute classical error metrics between X and Y_pred as returned
        by pred_quantiles().

        Call it from your dataprovider with a data facet to get regional values.
        """

        # Target column name
        target=self.dp.get_target()

        # General info and collection of IDs
        metrics={'classic:count': XY.shape[0]}

        import sklearn.metrics

        if XY.shape[0]<2:
            self.log(
                level=logging.WARNING,
                message='Data facet provided is empty or too small for this kind of metric computation. Skiping compute_trainsets_model_metrics_classical.'
            )
        elif self.estimator.is_classifier():
            proba_class_col=f'estimation_class_{self.dp.proba_class_index}'

            metrics=dict(
                **metrics,
                **{
                    # Collection of Metrics
                    'classic:ROC AUC':      sklearn.metrics.roc_auc_score(
                        y_true = XY[target],
                        y_score = Y_pred[proba_class_col]
                    ),
                    'classic:Average Precision Score': sklearn.metrics.average_precision_score(
                        y_true = XY[target],
                        y_score = Y_pred[proba_class_col]
                    ),
                }
            )
        else:
            metrics=dict(
                **metrics,
                **{
                    # Collection of Metrics
                    'classic:RMSE':           sklearn.metrics.mean_squared_error(
                        y_true = XY[target],
                        y_pred = Y_pred,
                        squared = False
                    ),

                     'classic:Mean Absolute Error':   sklearn.metrics.mean_absolute_error(
                        y_true = XY[target],
                        y_pred = Y_pred
                    ),

                    'classic:Mean Absolute Percentage Error': sklearn.metrics.mean_absolute_percentage_error(
                        y_true = XY[target],
                        y_pred = Y_pred
                    ),

                    # Med(|(ŷ-y)÷y|)
                    'classic:Median Absolute Percentage Error':  Model.median_ape(
                        y_true = XY[target],
                        y_pred = Y_pred
                    ),

                    # Med((ŷ-y)÷y)
                    'classic:Median Percentage Error':  Model.median_pe(
                        y_true = XY[target],
                        y_pred = Y_pred
                    ),

                    # https://en.wikipedia.org/wiki/Mean_percentage_error
                    # [ ∑(ŷ-y)÷y ] ÷ n
                    'classic:Mean Percentage Error': ((Y_pred-XY[target])/XY[target]).mean(),
                }
            )

        return metrics



    def compute_global_model_metrics_feature_importance_from_estimator(self,**args) -> dict:
        """
        Extract feature importance from estimator (NGBoost) and return a dict
        suitable for R3 metrics framework.

        Resulting dict will have this layout:

        {
            'feature importance:estimator:building_year:loc:importance': 0.0710
            'feature importance:estimator:building_year:loc:rank': 7.0,

            'feature importance:estimator:building_year:scale:importance': 0.078
            'feature importance:estimator:building_year:scale:rank': 7.0,

            'feature importance:estimator:complex_fee_m2:loc:importance': 0.112
            'feature importance:estimator:complex_fee_m2:loc:rank': 2.0,

            'feature importance:estimator:complex_fee_m2:scale:importance': 0.13
            'feature importance:estimator:complex_fee_m2:scale:rank': 3.0
            ...
        }
        """

        mprefix='feature importance:estimator:'

        if self.dp.hollow:
            # Since hollow Models doesn’t have an internal estimator, can’t
            # retrieve feature importance. Return an empty dict.
            return dict()
        else:
            return (
                pandas.DataFrame(
                    dict(
                        features=self.dp.get_estimator_features_list(),
                        importance=self.estimator.bagging_members[0].feature_importances_,
                    )
                )
                .sort_values('importance',ascending=False)

                # Useless index
                .reset_index(drop=True)

                # Usefull new index, lets use as the feature importance rank
                .reset_index(names='rank')

                # Make rank start with 1, not 0
                .assign(
                    rank   = lambda table: table['rank']   + 1
                )

                # Data wrangling for easy transformation into a dict
                # needed by Xingu metrics framework
                .set_index('features')
                .stack()
                .reset_index()

                # Fabricate metric names
                .assign(
                    name    = lambda table: mprefix + table.features + ':' + table.level_1
                )

                .drop(columns=['features','level_1'])

                .rename(columns={0: 'value_number'})
                .set_index('name')

                # Make it a dict() as Xingu needs for its DB
                .value_number
                .to_dict()
            )



    def render_global_model_plots_feature_importance_from_estimator(self):
        return (
            {
                'Feature Importance': (
                    pandas.DataFrame(
                        dict(
                            features = self.dp.get_estimator_features_list(),
                            importance = self.estimator.bagging_members[0].feature_importances_,
                        )
                    )
                    .sort_values('importance',ascending=True)

                    # Useless index
                    .set_index('features')

                    .plot
                    .barh()
                    .get_figure()
                )
            } if self.dp.hollow==False else None
        )


    def median_ape(y_true: pandas.Series, y_pred: pandas.Series) -> float:
        # Med(|(ŷ-y)÷y|)
        return ((y_true - y_pred) / y_true).abs().median()



    def median_pe(y_true: pandas.Series, y_pred: pandas.Series) -> float:
        # Med(|(y-ŷ)÷ŷ|)
        return ((y_pred - y_true) / y_true).median()



    ######################################################################################
    ##
    ## Object save and load in pickle format
    ##
    ######################################################################################



    def _handle_xz(file_obj, mode):
        """
        Compressor handler for Pathlib when file extension is .xz
        """
        preset=None
        if 'w' in mode:
            preset=9

        return lzma.LZMAFile(filename=file_obj, mode=mode, format=lzma.FORMAT_XZ, preset=preset)



    def save(self, path: str=None, dvc_path: str=None, compress=True):
        """
        Save pickle of this object. To be used later with Model.load().

        Args:
            path: S3 or filesystem folder path to save a PKL with standard name.
            dvc_path: local filesystem folder, usually same as path, to save model and
                wait for a commit to DVC.
        """


        resolved_path=self.get_config('TRAINED_MODELS_PATH', default=path)
        dvc_resolved_path=self.get_config('DVC_TRAINED_MODELS_PATH', default=dvc_path)

        if resolved_path.startswith('s3://'):
            import s3path
            resolved_path=s3path.S3Path.from_uri(resolved_path)
        else:
            resolved_path=pathlib.Path(resolved_path).resolve()

        filename_tpl  = '{dataprovider} • {time} • {full_train_id}.pkl'
        if compress: filename_tpl += '.xz'

        filename=filename_tpl.format(
            dataprovider   = self.dp.id,
            full_train_id  = self.get_full_train_id(),
            time           = Model.time_fs_str(self.trained)
        )

        target = resolved_path / filename

        if str(target).startswith('s3://'):
            target=urllib.parse.unquote(target.as_uri())

        self.log(
            'Serialized trained model object to {target}'.format(
                target=target
            )
        )

        import smart_open

        smart_open.register_compressor('.xz', Model._handle_xz)

        # Write to object storage or maybe filesystem
        with smart_open.open(target, mode="wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        if dvc_resolved_path:
            # DVC always, ALWAYS, works locally
            dvc_resolved_path=pathlib.Path(dvc_resolved_path).resolve()

            if dvc_resolved_path != resolved_path:
                # Will save a copy to be handled by DVC.
                # Later, externally, in a shell script, somebody has to do
                # a `dvc add` to include file in DVC.

                target = dvc_resolved_path / filename

                self.log(
                    'Serialize another copy of trained model for DVC to {target}'.format(
                        target=urllib.parse.unquote(target.as_uri())
                    )
                )
                with smart_open.open(urllib.parse.unquote(target.as_uri()), "wb") as f:
                    pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


            self.dvc_track(
                str(dvc_resolved_path),
                filename,
                "chore(serialized estimator)"
            )



    def load(dp_id: str, train_session_id: str='*', train_id: str='*', train_or_session_id_match_list: list=None, as_of: str='*', path: str=None):
        """
        Loads a pickle of a Model object. This is intentionally a static function. Don't
        call it with self.load(), but Model.load().

        On storage, pre-trained objects have file names as:

            ads_cur • 2021.10.07-14.03.27 • loud-box∶ragged-pedestal.pkl.xz

        Where
            - “ads_cur” is the DataProvider ID
            - “loud-box∶ragged-pedestal” is the train_session_id∶train_id

        If only dp_id is passed, last trained model of that DP will be loaded.

        If a train_id or train_session_id is also passed, chronology is ignored and that
        specific train is loaded.

        If a list is passed to train_and_session_id_match_list, tries to find any of the
        items in list as a term in file names.

        If a time and date is passed on as_of, the most current PKL at that time is
        selected.

        It doesn’t make sense to use train_and_session_id_match_list with train_id or
        train_session_id together.

        Args:
            dp_id: the ID of a DataProvider that was used to train a Model model. Will
            search storage for the most recent PKL file associated with this ID.

            train_id: load a specific train_id.

            path: filesystem or object storage (S3) path containing saved PKLs.
        """

        filename_template='{dataprovider_id} • {time} • {full_train_id}.pkl*'



        if path:
            resolved_path=path
        else:
            resolved_path=ConfigManager().get('TRAINED_MODELS_PATH', default=path)

        if not resolved_path:
            raise Exception("Don’t know where to load pre-trained models from. Set TRAINED_MODELS_PATH.")

#         self.log(f'Pre-trained model storage is {resolved_path}')

        if resolved_path.startswith('s3://'):
            import s3path
            resolved_path=s3path.S3Path.from_uri(resolved_path)
        else:
            resolved_path=pathlib.Path(resolved_path).resolve()

        if train_session_id!='*' and train_id!='*':
            # Have IDs for all
            full_train_id=f'{train_session_id}∶{train_id}'
        elif train_session_id=='*' and train_id=='*':
            # No IDs provided
            full_train_id='*'
        else:
            # Have IDs for any.
            # Result will be {train_session_id}* or *{train_id}
            full_train_id=f'{train_session_id}{train_id}'

        filename=filename_template.format(
            dataprovider_id    = dp_id,
            time               = '*',
            full_train_id      = full_train_id
        )

        # Get list of files in the path that match our filename glob
        # print(f'Searching for {resolved_path/filename}')
        available=[f for f in list(resolved_path.glob(filename)) if '.dvc' not in str(f)]

        available.sort()

        best=None
        model_file=None

        # Prepare filter by a list of train or session IDs to match
        if isinstance(train_or_session_id_match_list, list):
            for s in train_or_session_id_match_list:
                for i in range(len(available)-1,-1,-1):
                    if s in str(available[i]):
                        best=i
                        break
                if best is not None:
                    break

        if best is None:
            # Prepare filter by time
            if (isinstance(as_of,str) and as_of != '*') or (isinstance(as_of,pandas.Timestamp) or isinstance(as_of,datetime.datetime) or isinstance(as_of,datetime.date)):
                # Try to convert string to datetime
                as_of=pandas.to_datetime(as_of).to_pydatetime()

                # Extract only time from filenames
                times=[
                    datetime.datetime.strptime(
                        # Detect the datetime part
                        re.sub(r'.* • (.*) •.*',r'\1',x.name),
                        '%Y.%m.%d-%H.%M.%S'
                    ) for x in available
                ]

                # Now find the filename closer to as_of
                best=None
                for t in range(len(times)):
                    if times[t]<=as_of:
                        best=t
                    else:
                        break

        if best:
            model_file=available[best]
        elif len(available)>0:
            # Get last
            model_file=available[-1]
        else:
            raise FileNotFoundError(f'No pre-trained model for ‘{dp_id} • {train_session_id}:{train_id}’ to load. Search folder is ‘{resolved_path}’')

        import smart_open

        smart_open.register_compressor('.xz', Model._handle_xz)

        if str(model_file).startswith('s3://'):
            model_file=urllib.parse.unquote(model_file.as_uri())

        with smart_open.open(model_file, 'rb') as f:
            try:
                bob=pickle.load(f)
            except Exception as e:
                logging.error('Failed loading pre-trained model from {}. Error details follow.'.format(model_file))
                raise e

        return bob



    def load_pre_req_model(self):
        if (
            hasattr(self.dp, 'pre_req') and (
                not hasattr(self.dp, 'pre_req_model') or (
                    hasattr(self.dp, 'pre_req_model') and
                    self.dp.pre_req_model is None
                )
            )
        ):
            # DataProvider needs an external Model model to do feature engineering.
            # Search or load it here.
            self.dp.pre_req_model={}
            if self.coach:
                # Search for it in the coach RAM
                for pr in self.dp.pre_req:
                    self.log(f'Using coach’s pre-loaded models for {pr}')
                    self.dp.pre_req_model[pr] = self.coach.find_model_for_dp(pr)
                    if self.dp.pre_req_model[pr]:
                        self.log(f'Using pre-req in RAM {self.dp.pre_req_model[pr]}')


            if len(self.dp.pre_req_model.keys()) == 0:
                for pr in self.dp.pre_req:
                    try:
                        self.dp.pre_req_model[pr] = Model(dp=pr, trained=True)
                        self.log(f'Loaded pre-req from storage {self.dp.pre_req_model[pr]}')
                    except FileNotFoundError as e:
                        raise(f"Pre-trained Model for «{pr}» DataProvider not found in storage. It is a requirement for this model.")



    #####################################################################
    ##
    ## Operational procedures
    ##
    #####################################################################



    def cleanup(self):
        """
        Delete all internal data used for training and batch predict
        """

            # sets
            # sets_estimations
        attr="""
            batch_predict_data
            batch_predict_estimations
            batch_predict_time
        """.split()

        for a in attr:
            if hasattr(self, a):
                self.log(f"Flushing internal data: {a}")
                delattr(self,a)



    def get_config(self, config_item: str, default=ConfigManager.undefined, cast=ConfigManager.undefined):
        """
        Return best configuration found for a certain config_item.
        Usually configuration come from, in this order of priority:
        1. From what was set in the constructor
        2. From the coach, if we are running under the supervision of a coach
        3. From an environment variable
        4. Default value
        """

        if self.coach is None:
            return ConfigManager().get(config_item, default=default, cast=cast)
        else:
            return self.coach.get_config(config_item, default=default, cast=cast)



    def log(self, message, level=logging.INFO):
        self.logger.log(
            level,
            'train_session_id={train_session_id}, train_id={train_id}, {dp}: {message}'.format(
                train_session_id=self.train_session_id,
                train_id=self.train_id,
                dp=str(self.dp),
                message=message
            )
        )



    def log_train_status(self, status: str):
        """
        Log some status to DB training_status table
        """

        base=dict(
            train_session_id   = self.train_session_id,
            train_id           = self.train_id,
            dataprovider_id    = self.dp.id,
        )

        attr=list()

        with self.coach.get_db_connection().begin() as conn:
            if status == 'train_dataprep_start':
                # train_dataprep_start is the first step of a training session, so
                # take the chance to register also the overall train information

                attr += """
                    user_name
                    host_name
                    git_branch
                    git_commit
                    github_actor
                    github_workflow
                    github_run_id
                    github_run_number
                    x_estimator_features
                """.split()

            elif status == 'train_hyperhandle_end':
                # Capture some information related to hyperparameters optimization and handling

                attr += """
                    hyperparam
                """.split()

            elif status == 'train_fit_end':
                # Capture some information after training the actual estimator

                attr += """
                    estimator
                """.split()

            rows=list()
            for a in attr:
                value=None

                if a=='x_features' and len(self.dp.get_features_list())>0:
                    value=json.dumps(self.dp.get_features_list())
                elif a=='hyperparam':
                    value=json.dumps(self.algo_params)
                elif hasattr(self,a) and getattr(self,a) is not None:
                    value=str(getattr(self,a))
                else:
                    # Do not append to rows
                    continue

                rows.append(
                    dict(
                        **base,
                        **dict(
                            attribute  = a,
                            value      = value
                        )
                    )
                )

            if len(rows)>0:
                conn.execute(self.coach.tables['training_attributes'].insert().values(rows))

            conn.execute(
                self.coach.tables['training_steps'].insert().values([
                    dict(
                        **base,
                        **dict(
                            time       = round(datetime.datetime.utcnow().timestamp()),
                            status     = status
                        )
                    )
                ])
            )



    def fit_and_batch_predict(self):
        self.fit()
        self.batch_predict()



    def get_full_train_id(self):
        if self.train_session_id:
            return f'{self.train_session_id}∶{self.train_id}'
            # return f'{self.train_session_id},{self.train_id}'
        else:
            return self.train_id



    def dvc_track(self, path: str, file: str, message: str):
        if not hasattr(self,'dvc_tracked'):
            self.dvc_tracked={}

        if path not in self.dvc_tracked:
            self.dvc_tracked[path]=[]

        self.dvc_tracked[path].append((file,message))

        self.log(f'Added {file} to be tracked by DVC')



    def dvc_commit(self):
        """
        Issue DVC and Git commands to commit tracked files to version control.
        """
        if hasattr(self,'dvc_tracked'):
            self.log('Commit to Git and DVC')
            dvc_shell = []
            file_level_messages = ['File level messages:']

            # Template for our multi-line super-detailed commit message
            message='\n'.join([
                "build(estimator): {dp} • {train_session_id}∶{train_id}",
                '',
                "{signature}",
                '',
                "{file_level_messages}"
            ])

            for path in self.dvc_tracked:
                dvc_shell.append(f'cd "{path}"')
                for file in self.dvc_tracked[path]:
                    dvc_shell.append(f'dvc add "{file[0]}"')
                    dvc_shell.append(f'git add "{file[0]}.dvc"')
                    file_level_messages.append(f"   {file[0]}: {file[1]}")

            import yaml

            message=message.format(
                dp=self.dp.id,
                train_session_id=self.train_session_id,
                train_id=self.train_id,
                signature=yaml.dump(self.signature()),
                file_level_messages='\n'.join(file_level_messages)
            )

            dvc_shell.append(f'git commit -F - <<END\n{message}\nEND\necho')
            dvc_shell+=['git push','dvc push']

            self.log(
                level=logging.DEBUG,
                message='Commit to DVC as:\n{script}'.format(
                    script=';\n'.join(dvc_shell)
                )
            )

            dvc_shell=';\n'.join(dvc_shell)
            if not self.get_config('DRY_RUN', default=False, cast=bool):
                return_code=os.system(dvc_shell)

                if return_code == 0 and self.coach.git_repo:
                    self.git_artifact_commit=self.coach.git_repo.head.target.hex

                if return_code != 0:
                    raise OSError(return_code, f'Following external command failed: {dvc_shell}')

            del self.dvc_tracked



    def save_sets_cache(self):
        cache_path=self.get_config('DATSOURCE_CACHE_PATH', default=None)
        if cache_path:
            if cache_path.startswith('s3://'):
                import s3path

                # s3fs needs to be imported otherwise pyathena somehow grabs internal calls and fails
                import s3fs

                cache_pypath=s3path.S3Path.from_uri(cache_path)
            else:
                cache_pypath=pathlib.Path(cache_path).resolve()

            cache_template="cache • {context} • {dp} • {part}"

            for part in self.sets:
                cache_file_name_prefix=cache_template.format(
                    context    = self.context,
                    dp         = self.dp.id,
                    part       = part,
                )

                cache_file_name = cache_file_name_prefix + '.parquet'

                cache_file = cache_pypath / cache_file_name

                if cache_path.startswith('s3://'):
                    cache_file_uri=urllib.parse.unquote(cache_file.as_uri())

                self.log(f'Saving part on {cache_file_uri}')
                self.sets[part].to_parquet(cache_file_uri, compression='gzip')



    def data_source_to_data_from_url(self, url: str, params: dict=dict()) -> pandas.DataFrame:
        """
        The url parameters points to a local file, http://... or s3://...
        resource.

        This method will use pandas.read_csv(), pandas.read_parquet() or
        pandas.read_json() depending on text found in the URL.

        The params dict will be passed to the pandas method and might have any
        parameters accepted by these methods.
        """
        token_map=dict(
            json     = pandas.read_json,
            parquet  = pandas.read_parquet,
        )

        # Find correct Pandas method we´ll use.
        # Start with a default to pandas.read_csv()
        pandas_method = pandas.read_csv
        for token in token_map.keys():
            if token in url:
                pandas_method = token_map[token]
                break

        self.log(
            level=logging.DEBUG,
            message='Using {method} to retrieve dataset from {url}'.format(
                url = url,
                method = pandas_method
            )
        )

        return pandas_method(url, **params) if params else pandas_method(url)



    def data_source_to_data_from_query(self, query: str, con_nickname: str) -> pandas.DataFrame:
        # Remove excessive indenting
        query_text = textwrap.dedent(query)

        self.log(
            level=logging.INFO,
            message='Retrieving dataset from «{source}»:\n{query}'.format(
                source     = con_nickname,
                query      = textwrap.indent(query_text,'   ')
            )
        )

        # Hit the database
        with self.coach.get_db_connection(con_nickname).connect() as db:
            table = pandas.read_sql_query(
                sql = query_text,
                con = db
            )

        return table



    def data_source_to_data(self, sourceid: str, data_source: dict) -> pandas.DataFrame:
        """
        data_source is one entry of a complex DataProvider.train_data_sources or
        DataProvider.batch_predict_data_sources

        data_source might have following structures:

            dict(
                source='db_nickname' | 'xingu',
                query="SELECT ..."
            )

        OR

            dict(
                url="file:/..." | "http://..." | "s3://..."
            )

        sourceid is the name of this source, as it appears in the DP's dict.

        This method is called in parallel by data_sources_to_data().
        """

        """
        - Check for cache path
        - Compute unique hash for datasource
        - If cache exists, read_parquet()
        - If no cache, get data from source (URL, DB etc)
        - Save cache with unique hash
        - Return a dataframe
        """

        # Initialize to a non-sense value what we are going to return
        df=None

        if data_source is None:
            return df

        cache_path=self.get_config('DATASOURCE_CACHE_PATH', default=None)
        dvc_cache_path=self.get_config('DVC_DATASOURCE_CACHE_PATH', default=None)

        if cache_path is None:
            self.log(
                level=logging.WARNING,
                message="DATASOURCE_CACHE_PATH is not set, which is good for production environment. But it will make you access the (potentially slow and expensive) data source on every run."
            )

        cache_template="cache • {context} • {dp} • {sourceid} • {signature}"

        # Iterate over valid keys until we get something
        valid_keys = ['query', 'url']
        for key_type in valid_keys:
            query_text = data_source.get(key_type)
            if query_text is not None:
                break

        # Compute a unique hash for the SQL query text, for cache management purposes
        if cache_path or dvc_cache_path:
            # Compute cache file name using a signature from SQL query text
            cypher = hashlib.shake_256()
            cypher.update(query_text.encode('UTF-8'))
            signature = cypher.hexdigest(10)

            if self.context not in self.train_queries_signatures:
                self.train_queries_signatures[self.context]=dict()
            self.train_queries_signatures[self.context][sourceid]=signature

            cache_file_name_prefix=cache_template.format(
                context    = self.context,
                dp         = self.dp.id,
                sourceid   = sourceid,
                signature  = signature
            )

            cache_file_name=cache_file_name_prefix + '.parquet'

        if cache_path:
            # Lets try cache first

            if cache_path.startswith('s3://'):
                import s3path
                cache_pypath=s3path.S3Path.from_uri(cache_path)
            else:
                cache_pypath=pathlib.Path(cache_path).resolve()

            cache_file = cache_pypath / cache_file_name

            cache_file_uri=cache_file
            if cache_path.startswith('s3://'):
                cache_file_uri=urllib.parse.unquote(cache_file.as_uri())

            # Check if we have a file with this name
            if cache_file.is_file():
                # We have a cache hit. Use it.

                try:
                    df=pandas.read_parquet(cache_file_uri)
                except Exception:
                    # Cache is invalid and throws something like ArrowInvalid,
                    # so ignore and grab data from datasource again.
                    self.log(
                        'Invalid cache for «{source}» on {context}, looked for in file {cache_file}. Retrieving data from remote data source.'.format(
                            source      = sourceid,
                            cache_file  = cache_file_uri,
                            context     = self.context
                        ),
                        level=logging.WARNING
                    )
                else:
                    self.log(
                        'Using cache from {cache_file} instead of remote source for «{source}» on {context}'.format(
                            source      = sourceid,
                            cache_file  = cache_file_uri,
                            context     = self.context
                        )
                    )
            else:
                # No cache. Retrieve data from remote source and make cache.

                self.log(
                    'No cache for «{source}» on {context}, looked for in file {cache_file}. Retrieving data from remote data source.'.format(
                        source      = sourceid,
                        cache_file  = cache_file_uri,
                        context     = self.context
                    )
                )

        if df is None:
            # No success with cache so far.
            # Get data from original source.

            if key_type == 'query':
                df = self.data_source_to_data_from_query(
                    query_text,
                    data_source['source']
                )
            elif key_type == 'url':
                df = self.data_source_to_data_from_url(
                    query_text,
                    params=data_source['params'] if 'params' in data_source else dict()
                )

            self.log(
                'Data for «{source}» on {context} has shape {rows}×{cols}.'.format(
                    source      = sourceid,
                    context     = self.context,
                    rows        = df.shape[0],
                    cols        = df.shape[1],
                )
            )

            if cache_path:
                self.log(f'Making cache on {cache_file_uri}')
                with cache_file.open('bw') as f:
                    df.to_parquet(f, compression='gzip')

            if dvc_cache_path:
                if cache_path is None or dvc_cache_path!=cache_path:
                    # Need another copy for DVC
                    dvc_cache_file_name=pathlib.Path(
                        dvc_cache_path,
                        cache_file_name
                    ).resolve()

                    self.log(f'Making cache copy for DVC on {dvc_cache_file_name}')
                    df.to_parquet(
                        dvc_cache_file_name,
                        compression='gzip'
                    )

                self.dvc_track(
                    dvc_cache_path,
                    cache_file_name,
                    "chore(query cache)"
                )

        return df



    def data_sources_group_to_data(self, group: str, data_sources: pandas.DataFrame) -> dict:
        """
        data_sources is a DataFrame with data sources of a single group,
        like this:

        |   sequence | name           | query    | source      | url      | params          |
        |-----------:|:---------------|:---------|:------------|:---------|:----------------|
        |          3 | publico_202111 | SELECT … | my_datalake | nan      | nan             |
        |         25 | book_202111    | SELECT … | my_datalake | nan      | nan             |
        |         47 | url__202111    | nan      | nan         | https:// | {'param1': 1, … |
        |         69 | publico_202111 | SELECT … | my_datalake | nan      | nan             |
        |         91 | book_202111    | SELECT … | my_datalake | nan      | nan             |
        |        113 | safra_202111   | SELECT … | my_datalake | nan      | nan             |

        This method will iterate over each entry sequentially (not in
        parallel) and call data_source_to_data() to each.

        Returns a dict of DataFrames, as:

            dict(
                publico_202111 = pandas.DataFrame(…),
                book_202111    = pandas.DataFrame(…),
                …
            )
        """

        results = dict()

        for (sequence,datasource) in data_sources.iterrows():
            entry = datasource.to_dict()

            if pandas.isna(entry['query']):
                del entry['query']
                del entry['source']
            elif pandas.isna(entry['url']):
                del entry['url']
                del entry['params']

            name = entry['name']
            del entry['name']

            results[name] = self.data_source_to_data(name, entry)

        self.log(f"Finished data extraction of the {data_sources.shape[0]} entries in group {group}")

        return results



    def data_sources_to_data(self, data_sources: dict=None) -> dict:
        """
        This method is called by Model.fit() right after
        DataProvider.get_dataset_sources_for_train() or
        DataProvider.get_dataset_sources_for_batch_predict() returns a
        dict with queries. These DP methods usually pre-process content
        of DP.train_data_sources and DP.batch_predict_data_sources.
        DataProvider’s default implementation simply returns the content
        of these dicts, without pre-processing.

        This method simply runs in parallel multiple
        Model.data_source_to_data(), which is the method that really
        handles all database affairs, parquet cache, file naming etc.

        DataProvider.get_dataset_sources_for_train() must return a dict (that
        is passed to this method) that looks like:

            dict(
                df1=dict(
                    source='db_nickname' | 'xingu',
                    query="SELECT ..."

                    group='group_name' # (optional)
                ),

                df2=dict(
                    url="file:/..." | "http://..." | "s3://..."

                    params=dict(...) # (optional, passed to read_csv, read_parquet, read_json)
                    group='group_name' # (optional)
                ),

                df3=dict(...)
            )

        Entries will be grouped by the 'group' key and each group will be
        executed in parallel. Entries under same group will be executed in
        sequence. Entries without group designation will receive their own
        single-entry groups, thus will be executed in parallel with other
        groups. Since entries under same group are executed in sequence, groups
        are useful with queries that depend on temporary tables or other
        objects that will be generated by previous entries in the same group.

        Returns a dict with similar structure containing DataFrames with
        data returned by these queries. Ready to be passed to
        DataProvider.clean_data().

        Here is an example of returned dict:

            dict(
                df1=pandas.DataFrame(...),
                df2=pandas.DataFrame(...),
                df3=pandas.DataFrame(...),
            )
        """

        if data_sources is None:
            return None

        # Define how many parallel workers to execute
        max_workers=self.get_config('PARALLEL_DATASOURCE_MAX_WORKERS', default=0, cast=int)
        if max_workers == '' or max_workers == 0:
            max_workers=None

        # Prepare the data extraction plan
        data_extraction_plan = (
            pandas.DataFrame(data_sources).T

            # Handle the name of data source, which will become the key
            # to the resulting dataframe
            .reset_index()
            .rename(columns=dict(index='name'))

            # Order of data sources declared in the dict matters because
            # of dependencies. Handle it.
            .reset_index()
            .rename(columns=dict(index='sequence'))

            .assign(
                # Make sure we have all expected columns, even if they
                # were not given by the user. Code ahead would break if
                # columns simply do not exist, so fabricate them with
                # NaNs.
                query  = lambda table: table['query']  if 'query'  in table.columns else None,
                source = lambda table: table['source'] if 'source' in table.columns else None,
                url    = lambda table: table['url']    if 'url'    in table.columns else None,
                params = lambda table: table['params'] if 'params' in table.columns else None,


                # Group is a new feature. For backward compatibility we’ll
                # use the sequence number as a group name, so each data
                # source name will have its own group. In case we are
                # using groups but only in a few data sources, entries
                # without group names will get its sequence as group name.
                group=lambda table: (
                    (
                        table.group
                        if 'group' in table
                        else table.sequence
                    )
                    .combine_first(table.sequence)
                )
            )


            # Organize everything in the way we are going to process
            .sort_values('group sequence'.split())

            .set_index('group sequence'.split())

            # .reset_index(level=1)

            # Now we are ready to iterate over each group in parallel and fire each of
            # its data extraction in sequence.
        )

        self.log(
            (
                "Data extraction plan has {nentries} entries in {ngroups} "
                "groups. {nquery} are typical database queries, {nurl} "
                "are files or URLs. {parallel} groups will be executed "
                "in parallel, but entries on each group will be executed "
                "sequentially."
            ).format(
                nentries = data_extraction_plan.shape[0],
                nquery   = data_extraction_plan['query'].count(),
                nurl     = data_extraction_plan['url'].count(),
                parallel = "All" if max_workers is None else max_workers,
                ngroups  = len(
                    data_extraction_plan
                    .index
                    .get_level_values(level=0)
                    .unique()
                ),
            )
        )

        self.log(
            level=logging.DEBUG,
            message=(
                "Data extraction plan:\n" + (
                    data_extraction_plan[['name','source','url']]
                    .reset_index()
                    .to_markdown()
                )
            )
        )

        # Iterate over all data sources groups and fire
        # data_sources_group_to_data() for each.
        # PARALLEL_DATASOURCE_MAX_WORKERS defines how many groups will be
        # executed in parallel. Data sources inside a single group are
        # extracted sequentially, not in parallel.
        collected_data=dict()
        with concurrent.futures.ThreadPoolExecutor(
                        thread_name_prefix='data_sources_to_data',
                        max_workers=max_workers
                ) as e:

            # Submit all groups to execute in parallel
            tasks = {
                e.submit(
                    self.data_sources_group_to_data,
                    group,
                    data_extraction_plan.loc[group]
                ): group
                for group in data_extraction_plan.index.get_level_values(level=0).unique()
            }

            for task in concurrent.futures.as_completed(tasks):
                # For each group that finished processing, get all resulting
                # DataFrames at once
                collected_data.update(task.result())

        return collected_data



    def time_fs_str(time):
        time_tpl='%Y.%m.%d-%H.%M.%S'
        return time.strftime(time_tpl)



    def serialize_to_dict(self, include: set=None, exclude: set=None):
        """
        Return a dict version of this object, including only the attributes on
        include and excluding the attributes on exclude.

        Used by __getstate__() and signature()
        """
        all_attributes=set(self.__dict__.keys())
        interested=all_attributes

        if include:
            interested = all_attributes.intersection(include)

        if exclude:
            interested = interested-exclude

        return {k:self.__dict__[k] for k in interested}



    def __getstate__(self):
        """
        Be selective on what to pickle: exclude coach, logger and
        DataFrames used and generated throughout train and batch predict
        session.
        """

        attributes={
            # DataProvider
            'dp',

            # Train info
            'train_session_id',     'train_id',    'trained',
            'hyperopt_strategy',    'algo_params', 'train_queries_signatures',

            # Large stuff
            'estimator',

            # OS environment info
            'user_name',            'host_name',

            # Code info
            'git_branch',           'git_commit',

            # Trained artifacts info
            'git_artifact_commit',

            # GitHub Actions info
            'github_actor',         'github_workflow',
            'github_run_id',        'github_run_number',
        }

        state=self.serialize_to_dict(include=attributes)

        if self.get_config('DRY_RUN', default=False, cast=bool):
            # The large estimator object was asked to be excluded (for dev
            # purposes), so put an empty similar object inplace.
            state['estimator']=type(self.estimator)()

        return state



    def signature(self) -> dict:
        """
        Return a dict ready to be converted into YAML for inventory.yaml
        """
        attributes={
            # Train info
            'train_session_id',     'train_id',    'trained',
            'hyperopt_strategy',    'algo_params', 'train_queries_signatures',

            # OS environment info
            'user_name',            'host_name',

            # Code info
            'git_branch',           'git_commit',

            # Trained artifacts info
            'git_artifact_commit',

            # GitHub Actions info
            'github_actor',         'github_workflow',
            'github_run_id',        'github_run_number',
        }

        signature=self.serialize_to_dict(include=attributes)
        signature['dataprovider_id']=self.dp.id

        if hasattr(self.dp, 'pre_req_model'):
            signature['pre_req']={
                dp: self.dp.pre_req_model[dp].signature()
                    for dp in self.dp.pre_req_model
            }

        return signature



    def __repr__(self):
        template=(
            'Model(' +
                'dp={dp},' +
                ' trained={trained},' +
                ' train_session_id={train_session_id},' +
                ' train_id={train_id},' +
                ' hyperopt_strategy={hyperopt_strategy},' +
                ' estimator={estimator}' +
            ')'
        )

        return template.format(
            dp                   = self.dp,
            trained              = self.trained,
            train_session_id     = self.train_session_id,
            train_id             = self.train_id,
            hyperopt_strategy    = self.hyperopt_strategy,
            estimator            = self.estimator,
        )
