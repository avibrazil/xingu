import os
import os.path
import textwrap
import inspect
import pwd
import socket
import lzma
import randomname
import logging
import datetime
import re
import glob
import pathlib
import hashlib
import json
import concurrent.futures
import pickle
import yaml
import sqlalchemy
import numpy
import pandas as pd
import s3path
import smart_open
import urllib
import sklearn.metrics as sklm

from . import DataProvider
from . import Coach
from . import Estimator
from . import ConfigManager






class Model(object):


    ######################################################################################
    ##
    ## Train a Model object, including hyperparam optimization
    ##
    ######################################################################################



    def optimize_hyperparam(self):
        """
        Decide wether to compute or use previously computed hyperparameters.

        This is controlled by self.hyperopt_strategy, which can be:

        - self -- compute hyperparameters optimization
        - last -- search Model DB for last computed hyperparams for this DataProvider
        - dp -- use hyperparams as returned by self.dp.get_estimator_hyperparameters()
        - a train name -- load from Model DB the previously computed hyperparams used on
             a specific train_id
        """

        self.context='train_hyperopt'

        # Declare we have no clue of hyper-params
        self.hyperparam=None
        dbresult=None

        if self.hyperopt_strategy == 'self':
            # Lengthy estimator hyper-parameter optimization
            self.log_train_status('train_hyperopt_start')

            h=self.estimator.hyperparam_optimize(
                train     = self.sets['train'],
                val       = self.sets['val'],
                features  = self.dp.get_features_list(),
                target    = self.dp.get_target()
            )

            self.hyperparam=dict(
                source_train_id='self',
                hyperparam=h
            )

            self.log_train_status('train_hyperopt_end')


        elif self.hyperopt_strategy == 'dp':
            # Get from DataProvider

            hyperparam=self.dp.get_estimator_hyperparameters()

            if hyperparam is not None:
                self.hyperparam=dict(
                    source_train_id='dp',
                    hyperparam=hyperparam
                )


        elif self.hyperopt_strategy is not None:
            # Get from DB last computed hyper-parameters for this DP or a specific train_id.

            table=self.coach.tables['training_status']

            query=(
                table.select()
                    .where(
                        sqlalchemy.and_(
                            table.c.dataprovider_id==self.dp.id,
                            table.c.status=='train_hyperopt_end',
                        )
                    )
                    .order_by(table.c.time.desc())
                    .limit(1)
            )

            if self.hyperopt_strategy != 'last':
                # Include additional condition
                query=query.where(
                    sqlalchemy.or_(
                        table.c.train_id         == self.hyperopt_strategy,
                        table.c.train_session_id == self.hyperopt_strategy,
                    )
                )

            dbresult = query.execute().first()


            if dbresult is not None:
                # Import the JSON text into a dict
                self.hyperparam=json.loads(dbresult.hyperparam)

                if self.hyperparam['source_train_id']=='self':
                    # Give credit for the source of hyper-params
                    self.hyperparam['source_train_id']=dbresult.train_id

        # At this point we have hyperparam self-computed, or retrieved from DB, or None
        if self.hyperparam is None:
            # If still None, get defaults from estimator
            self.hyperparam=dict(
                source_train_id='default',
                hyperparam=self.estimator.hyperparam_exchange()
            )
        else:
            # If we have something, set it on estimator
            self.estimator.hyperparam_exchange(self.hyperparam['hyperparam'])



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

        self.train_id          = randomname.get_name()
        if self.coach:
            self.train_session_id  = self.coach.train_session_id

        self.context='train_dataprep'
        self.log_train_status('train_dataprep_start')

        # Get DP SQL queries
        self.log(message='Asking DataProvider for training queries')
        sources=self.dp.get_dataset_sources_for_train()

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
            message=f'Feature engineering returned columns: {list(train_data.columns)}'
        )

        # Give it a chance to post-process data before training
        train_data=self.dp.last_pre_process_for_train(train_data)

        # Train/test split
        self.log(message='DataProvider will split data')
        self.sets=self.dp.data_split_for_train(train_data)
        
        if self.sets is not None:
            # Convert the tuple into a meaningful dict
            self.sets=dict(
                train = self.sets[0],
                val   = self.sets[1],
                test  = self.sets[2]
            )

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
            self.log(message='Xingu will do some hyperparameter optimization')
            self.optimize_hyperparam()
            self.log(message=f'Hyperopt parameters: {self.hyperparam}', level=logging.DEBUG)


            self.context='train_fit'
            self.log_train_status('train_fit_start')

            self.log(message='Training an Estimator')
            self.estimator.fit(
                train     = self.sets['train'],
                val       = self.sets['val'],
                features  = self.dp.get_estimator_features_list(),
                target    = self.dp.get_target()
            )

            self.log_train_status('train_fit_end')

        self.trained=datetime.datetime.now(datetime.timezone.utc)
        # A chance for the DataProvider to manipulate the Model object or use
        # it to compute something before any PKL saving or metrics computation.
        self.dp.post_process_after_train(self)

        self.context=None



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
        sources=self.dp.get_dataset_sources_for_batch_predict()
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
        self.batch_predict_valuations=self.pred_quantiles(self.batch_predict_data)

        # Record time of estimation
        self.batch_predict_time = datetime.datetime.now(datetime.timezone.utc)

        # Standardize output
        self.batch_predict_valuations.rename(columns={'scale':'sigma'}, inplace=True)

        self.context=None



    ######################################################################################
    ##
    ## Predict methods
    ##
    ######################################################################################


    def generic_predict(self, data: pd.DataFrame, quantiles: bool=False) -> pd.DataFrame:
        """
        Calls estimator.pred_quantiles() (quantiles==True) or
        estimator.pred_dist().
        
        Return Y_pred with p_* quantiles (quantiles==True) or just μ and σ.
        """
        if quantiles:
            methods=dict(
                     predict = None if self.dp.hollow else getattr(self.estimator, 'pred_quantiles'),
                 pre_predict = getattr(self.dp, 'pre_process_for_pred_quantiles'),
                post_predict = getattr(self.dp, 'post_process_after_pred_quantiles'),
            )
        else:
            methods=dict(
                     predict = None if self.dp.hollow else getattr(self.estimator, 'pred_dist'),
                 pre_predict = getattr(self.dp, 'pre_process_for_pred_dist'),
                post_predict = getattr(self.dp, 'post_process_after_pred_dist'),
            )

        # Do whatever pre-processing the DP wants to do
        prepared=methods['pre_predict'](X=data.copy(), model=self)
        
        
        # Call predict method only if we are not hollow
        Y_pred=methods['predict'](
            prepared[self.dp.get_estimator_features_list()]
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

    

    def pred_dist(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        The most basic estimation service.

        Returns a DataFrame indexed by unit_id containing loc, scale and score
        of estimation (in case of multiple models).

        Input data must contain at least self.dp.get_estimator_features_list()
        columns and has to be pre-feature-engineered, including for pre-req
        models. In other words, data must have columns ready for predict methods
        of this Model and also for its pre-reqs.
        
        Use your DP’s feature_engineering_for_predict() method before calling
        pred_dist().
        """

        self.log(f'Estimating price probability distribution for {data.shape[0]} data points')
        
        # 1. Pre-process
        prepared=self.dp.pre_process_for_pred_dist(X=data.copy(), model=self)
        
        # 2. Predict
        Y_pred = None
        if not self.dp.hollow:
            Y_pred=self.estimator.pred_dist(
                prepared[self.dp.get_estimator_features_list()]
            )
        
        # 3. Post-process
        return (
            self.dp.post_process_after_pred_dist(
                X         = prepared,
                Y_pred    = Y_pred,
                model     = self
            )
            
            # Reorder to match input
            .reindex(prepared.index)
        )



    def pred_quantiles(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        The most complete estimation service.

        Returns a DataFrame indexed by unit_id containing p_05..p_95, loc
        (μ, as p_50), sigma and score of estimation (in case of multiple
        models).

        Input data must contain at least self.dp.get_estimator_features_list()
        columns and has to be pre-feature-engineered, including for pre-req
        models. In other words, data must have columns ready for predict methods
        of this Model and also for its pre-reqs.
        
        Use your DP’s feature_engineering_for_predict() method before calling
        pred_quantiles().
        """

        self.log(f'Estimating discrete quantiles of price probability distribution for {data.shape[0]} data points')

        # 1. Pre-process
        prepared=self.dp.pre_process_for_pred_quantiles(X=data.copy(), model=self)
        
        # 2. Predict
        Y_pred = None
        if not self.dp.hollow:
            Y_pred=self.estimator.pred_quantiles(
                prepared[self.dp.get_estimator_features_list()]
            )

        # 3. Post-process
        return (
            self.dp.post_process_after_pred_quantiles(
                X         = prepared,
                Y_pred    = Y_pred,
                model     = self
            )
            
            # Reorder to match input
            .reindex(prepared.index)
        )



    ######################################################################################
    ##
    ## Post-Fit and post-Batch Predict procedures
    ##
    ######################################################################################



    def save_train_sets(self):
        if self.sets is not None:
            self.context='train_savesets'
            self.log_train_status('train_savesets_start')


            # Record sets in DB
            for part in self.sets.keys():
                self.log(f'Saving {part} dataset to DB')

                target=self.dp.get_target()
                if target in self.sets[part].columns:
                    df=self.sets[part][[target]].rename(columns={target: 'target'})
                    df.index.rename('unit_id', inplace=True)
                    df.reset_index(inplace=True)
                    df['set']                = part
                    df['dataprovider_id']    = self.dp.id
                    df['train_id']           = self.train_id
                    df['train_session_id']   = self.train_session_id

                    self.log(level=logging.DEBUG, message=f'{part}: {df.shape[0]}×{df.shape[1]}')

                    df.to_sql(
                        self.coach.tables['sets'].name,
                        if_exists='append',
                        index=False,
                        con=self.coach.get_db_connection('xingu')
                    )

            self.log_train_status('train_savesets_end')



    def save_batch_predict_valuations(self):
        self.log('Save predicted prices to DB::valuations table')

        to_save=self.batch_predict_valuations.copy(deep=False)

        if hasattr(self, 'batch_predict_valuations'):
            # Tag valuations with context
            to_save['time']              = round(self.batch_predict_time.timestamp())
            to_save['train_id']          = self.train_id
            to_save['train_session_id']  = self.train_session_id
            to_save['dataprovider_id']   = self.dp.id

            # Handle score later, as a metric
            columns=list(set(to_save.reset_index().columns)-{'score'})

            to_save.reset_index()[columns].to_sql(
                name            = self.coach.tables['valuations'].name,
                if_exists       = 'append',
                index           = False,
#                 con=self.get_config('XINGU_DB')
                con=self.coach.get_db_connection('xingu')
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
    # • compute_valuation_metrics_{NAME}(self, XY, Y_pred)
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
    #         self.compute_model_metrics() calls:
    #             self.compute_trainsets_model_metrics() calls:
    #                 All self.compute_trainsets_model_metrics_{NAME}()
    #                 All self.dp.compute_trainsets_model_metrics_{NAME}()
    #             self.compute_global_model_metrics() calls:
    #                 All self.compute_global_model_metrics_{NAME}()
    #                 All self.dp.compute_global_model_metrics_{NAME}()
    #     self.save_valuations_metrics() calls:
    #         self.compute_valuation_metrics() calls:
    #             All self.compute_valuation_metrics_{NAME}()
    #             All self.dp.compute_valuation_metrics_{NAME}()
    #



    def get_metrics_computers(self, type: str='valuation') -> list:
        """
        Return a list of member functions that can compute metrics.

        Current known types are:

        • trainsets_model
        • global_model
        • valuation
        """

        methods=[]
        for method in inspect.getmembers(self, predicate=inspect.ismethod):
            if 'compute_' + type + '_metrics_' in method[0]:
                methods.append(method[1])

        return methods



    def compute_trainsets_model_metrics(self) -> dict:
        """
        Calls all compute_trainset_model_metrics_{NAME}() from Model and DataProvider
        to compute metrics for self.sets DataFrames. Y_pred will be computed
        with pred_quantiles().
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

                Y_pred=self.pred_quantiles(self.sets[part])

                for domain in methods.keys():
                    params=dict(
                        XY      = self.sets[part],
                        Y_pred  = Y_pred
                    )

                    # Add model object as parameter if this is DP-provided method
                    if domain == 'dp':
                        params.update({'model': self})

                    for train_metrics_computer in methods[domain]:
                        all_metrics[part].update(
                            train_metrics_computer(**params)
                        )

        self.log(f'Train metrics: {all_metrics}', level=logging.DEBUG)
        return all_metrics



    def compute_global_model_metrics(self, XY: pd.DataFrame, Y_pred: pd.DataFrame) -> dict:
        """
        Calls all compute_global_model_metrics_{NAME}() from Model and DataProvider.
        """

        methods = dict(
            model   = self.get_metrics_computers(type='global_model'),
            dp      = self.dp.get_metrics_computers(type='global_model')
        )

        metrics={}
        for domain in methods.keys():
            params=dict(
                XY      = XY,
                Y_pred  = Y_pred
            )

            # Add model object as parameter if this is DP-provided method
            if domain == 'dp':
                params.update({'model': self})

            for global_metrics_computer in methods[domain]:
                metrics.update(global_metrics_computer(**params))

        return metrics



    def compute_model_metrics(self) -> pd.DataFrame:
        """
        Calls all compute_trainset_model_metrics_{NAME}() and
        compute_global_model_metrics_{NAME}() from Model and DataProvider.

        All dict to DataFrame handling happens here.
        """

        self.log('Compute model metrics.')

        model_metrics = {}

        # Get metrics for train, test and val sets
        if hasattr(self, 'sets') and self.sets is not None:
            # Operate only if we are into a training session, only if we have actual data.
            model_metrics.update(self.compute_trainsets_model_metrics())
        else:
            self.log('No train data to compute model metrics.', level=logging.WARNING)

        # Get global metrics
        if hasattr(self, 'batch_predict_data') and hasattr(self, 'batch_predict_valuations'):
            # Operate only if we batch predict data and estimations
            model_metrics.update(
                {
                    'global': self.compute_global_model_metrics(
                        XY       = self.batch_predict_data,
                        Y_pred   = self.batch_predict_valuations
                    )
                }
            )
        else:
            self.log('No data to compute model global metrics.', level=logging.WARNING)


        # Now convert the model_metrics dict into the metrics DataFrame.
        # Separate numeric and textual values.
        metrics=None
        for part in model_metrics:
            match_real_numbers=r'((\+|-)?([0-9]+)(\.[0-9]+)?)|((\+|-)?\.?[0-9]+)'
            met=pd.Series(model_metrics[part])
            metricsdf=(
                # Start with only the numeric metrics
                pd.DataFrame(met[met.astype(str).str.match(match_real_numbers)]
                .rename('value_number'))

                # Add columns for textual metrics
                .join(
                    how='outer',
                    other=(
                        met[~met.astype(str).str.match(match_real_numbers)]
                        .rename('value_text')
                    )
                )

                # Make sure empty cells have null
                .fillna(pd.NA)
            )


            # Convert to appropriate type
            metricsdf['value_number']=pd.to_numeric(metricsdf['value_number'])

            # Make the metric name a regular column called 'name'
            metricsdf.index.rename('name', inplace=True)
            metricsdf.reset_index(inplace=True)

            metricsdf['set'] = part

            if metrics is None:
                metrics=metricsdf
            else:
                metrics=pd.concat([metrics,metricsdf])

        return metrics



    def save_model_metrics(self):
        self.context='model_metrics'
        self.log_train_status('model_metrics_start')

        metrics=self.compute_model_metrics()

        if metrics is not None:
            # Tag metrics with IDs of this training session
            metrics['time']               = round(self.batch_predict_time.timestamp())
            metrics['dataprovider_id']    = self.dp.id
            metrics['train_session_id']   = self.train_session_id
            metrics['train_id']           = self.train_id

            # Save metrics to DB
            self.log('Saving model metrics to DB')
            metrics.to_sql(
                name           = self.coach.tables['metrics_model'].name,
                if_exists      = 'append',
                index          = False,
#                 con=self.get_config('ROBSON_DB')
                con=self.coach.get_db_connection('xingu')
            )

        self.log_train_status('model_metrics_end')



    def compute_valuation_metrics(self) -> pd.DataFrame:
        """
        Calls all compute_valuation_metrics_{NAME}() from Model and DataProvider.

        Each method must return a DataFrame with following columns:
        - unit_id (has to be the index)
        - name -- name of metric, such as 'score'
        - value_number -- metric value if numeric, such as '0.97234'
        - value_text -- metric value if it is text, such as 'red'

        Methods being called will get parameter:
        - self.batch_predict_data as XY
        - self.batch_predict_valuations as Y_pred
        """

        self.log(f'Compute per-unit valuation metrics')

        metrics=None
        if hasattr(self, 'batch_predict_data') and hasattr(self, 'batch_predict_valuations'):
            methods = dict(
                model   = self.get_metrics_computers(type='valuation'),
                dp      = self.dp.get_metrics_computers(type='valuation')
            )

            for domain in methods.keys():
                for valuation_metrics_computer in methods[domain]:
                    params=dict(
                        XY      = self.batch_predict_data,
                        Y_pred  = self.batch_predict_valuations
                    )

                    if domain == 'dp':
                        params.update({'model': self})

                for valuation_metrics_computer in methods[domain]:
                    m1=valuation_metrics_computer(**params)

                    if metrics is None:
                        metrics=m1
                    elif m1 is not None:
                        metrics=pd.concat([metrics,m1])

            if metrics is not None:
                # Make sure index has correct name
                metrics.index.rename(self.batch_predict_data.index.name, inplace=True)

                # Drop rows where BOTH value_number AND value_text have NaN
                metrics.dropna(
                    subset={'value_number','value_text'}.intersection(metrics.columns),
                    how='all',
                    inplace=True
                )
            else:
                self.log('No computer methods for valuation metrics', level=logging.WARNING)

        else:
            self.log('No data to compute valuation metrics.', level=logging.WARNING)

        return metrics



    def save_valuation_metrics(self):
        """
        Trigger computation of valuation metrics and save them on DB.
        """

        self.log(f'Preparing for valuation metrics')

        metrics=None

        metrics=self.compute_valuation_metrics()
        self.log(f'Valuation metrics computed')

        if metrics is not None:
            # Tag metrics with some context
            metrics['time']               = round(self.batch_predict_time.timestamp())
            metrics['train_id']           = self.train_id
            metrics['train_session_id']   = self.train_session_id
            metrics['dataprovider_id']    = self.dp.id

            self.log(f'Save valuation metrics to DB')

            metrics.reset_index().to_sql(
                name         = self.coach.tables['metrics_valuation'].name,
                if_exists    = 'append',
                index        = False,
#                 con=self.get_config('ROBSON_DB')
                con=self.coach.get_db_connection('xingu')
            )



    def compute_and_save_metrics(self):
        self.save_model_metrics()
        self.save_valuation_metrics()



    def compute_valu__OBSOLETE__ation_metrics_score(self, XY: pd.DataFrame, Y_pred: pd.DataFrame) -> pd.DataFrame:
        # Score is only available if using more than 1 estimator
        if 'score' in Y_pred.columns:
            score_metric=Y_pred[['score']].rename(columns={'score': 'value_number'})
            score_metric['name']='score'

            return score_metric[['name','value_number']]



    def compute_global_model_metrics_value_per_meter(self, XY: pd.DataFrame, Y_pred: pd.DataFrame) -> dict:
        """
        Compute μ, median and σ of value per m² for input dataset.
        
        Call it from your dataprovider with a data facet to get regional values.
        """
        
        if XY.shape[0]==0:
            return {}
        
        POINT_ESTIMATE = self.estimator.POINT_ESTIMATE if self.estimator else self.dp.POINT_ESTIMATE
        
        metrics={            
            'classic:Mean value per m²':    Y_pred[POINT_ESTIMATE].mean(),

            'classic:Median value per m²':  Y_pred[POINT_ESTIMATE].median(),

            'classic:σ value per m²':       Y_pred[POINT_ESTIMATE].std(),
        }
        
        return metrics
    
    
    
    def compute_trainsets_model_metrics_classical(self, XY: pd.DataFrame, Y_pred: pd.DataFrame) -> dict:
        """
        Compute classical error metrics between X and Y_pred as returned
        by pred_quantiles().
        
        Call it from your dataprovider with a data facet to get regional values.
        """

        # Target column name
        target=self.dp.get_target()

        POINT_ESTIMATE = self.estimator.POINT_ESTIMATE if self.estimator else self.dp.POINT_ESTIMATE
        UPPER_BOUND = self.estimator.UPPER_BOUND if self.estimator else self.dp.UPPER_BOUND
        LOWER_BOUND = self.estimator.LOWER_BOUND if self.estimator else self.dp.LOWER_BOUND
        QUANTILES = self.estimator.QUANTILES if self.estimator else self.dp.QUANTILES
        
        if XY.shape[0]<2:
            self.log(
                level=logging.WARNING,
                message='Data facet provided is empty or too small for this kind of metric computation. Skiping compute_trainsets_model_metrics_classical.'
            )
            return {
                'classic:count':        XY.shape[0]
            }

        metrics={
            # General info and collection of IDs
            'classic:count':        XY.shape[0],

            # Collection of Metrics
            'classic:RMSE':                              sklm.mean_squared_error(
                y_true = XY[target],
                y_pred = Y_pred[POINT_ESTIMATE],
                squared = False
            ),

             'classic:Mean Absolute Error':              sklm.mean_absolute_error(
                y_true = XY[target],
                y_pred = Y_pred[POINT_ESTIMATE]
            ),

            'classic:Mean Absolute Percentage Error':    sklm.mean_absolute_percentage_error(
                y_true = XY[target],
                y_pred = Y_pred[POINT_ESTIMATE]
            ),

            # Med(|(ŷ-y)÷y|)
            'classic:Median Absolute Percentage Error':  Model.median_ape(
                y_true = XY[target],
                y_pred = Y_pred[POINT_ESTIMATE]
            ),

            # Med((ŷ-y)÷y)
            'classic:Median Percentage Error':  Model.median_pe(
                y_true = XY[target],
                y_pred = Y_pred[POINT_ESTIMATE]
            ),

            # https://en.wikipedia.org/wiki/Mean_percentage_error
            # [ ∑(ŷ-y)÷y ] ÷ n
            'classic:Mean Percentage Error':             (
                (
                    Y_pred[POINT_ESTIMATE]-XY[target]
                ) / XY[target]
            ).mean(),

            # [ ∑(p80-p20)÷p50 ] ÷ n
            'classic:Interval':                          (
                (
                    Y_pred[UPPER_BOUND]-Y_pred[LOWER_BOUND]
                ) / Y_pred[POINT_ESTIMATE]
            ).mean(),
        }

        quantile_columns = [q[0] for q in QUANTILES]
        dropme = list(set(Y_pred.columns) - set(quantile_columns))
        
        # For each p_*, compute proportion of error bigger than that p_*.
        # For example, if this number for p_20 is 0.4, means that 40% of p_20’s
        # numbers are above 20% of the expected correct price.
        calibration = (
            (
                Y_pred
                [quantile_columns]
                .drop(columns=dropme, errors='ignore')
                .gt(XY[target], axis=0)
                .mean()
            ) -
            pd.DataFrame(self.dp.QUANTILES).set_index(0)[1]
        )

        metrics['classic:Quantile Deviation'] = calibration.abs().mean()

        calibration.index = 'classic:Quantile Deviation ' + calibration.index

        metrics.update(calibration.to_dict())
        
        # Calculate percentiles for Absolute Percentage Error (APE)
        # Use numpy.percentile - percentile values from 0 to 100
        quantile_values = [q[1] for q in QUANTILES]

        percentile_ape = [numpy.percentile(
            numpy.abs(Y_pred[POINT_ESTIMATE] - XY[target])/XY[target], q=100*quantile_value
        ) for quantile_value in quantile_values]
        
        for idx, q in enumerate(quantile_columns):
            metrics[f'classic:Absolute Percentage Error {q}'] = percentile_ape[idx]
        
        return metrics



    def compute_global_model_metrics_OKR15p(self, XY: pd.DataFrame, Y_pred: pd.DataFrame, point='p_50') -> dict:
        """
        Compute proportion of units where estimation error is higher than 15% (OKR 15%).
        """

        # Target column name
        target=self.dp.get_target()

        self.log('Computing OKR 15% metric')

        # Get only the unit_ids which we know the true price
        known_price_index=XY[~(XY[target].isna())].index

        if XY.loc[known_price_index].shape[0]<2:
            self.log(
                level=logging.WARNING,
                message='Data facet provided is empty or too small for this kind of metric computation. Skiping compute_global_model_metrics_OKR15p.'
            )
            return {
                'OKR error > 15%:proportion': 0,
                'OKR error > 15%:count':      XY.loc[known_price_index].shape[0]
            }

        self.log(f'{XY.shape[0]} of {XY.loc[known_price_index].shape[0]} units have known price')

        # count(|y÷ŷ - 1| > 15%) ÷ n
        count_of_big_errors = (
            (
                XY.loc[known_price_index][target] /
                Y_pred.loc[known_price_index][point] -
                1
            )
            .abs() > (15/100)
        ).value_counts()

        count_of_big_errors = count_of_big_errors[True] if True in count_of_big_errors else 0

        metrics={
            'OKR error > 15%:proportion': count_of_big_errors / XY.loc[known_price_index].shape[0],
            'OKR error > 15%:count':      XY.loc[known_price_index].shape[0]
        }

        return metrics



    def compute_global_model_metrics_feature_importance_from_estimator(self, XY: pd.DataFrame, Y_pred: pd.DataFrame) -> dict:
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
                # Organize information in a DataFrame. God, we love pandas.DataFrame
                pd.DataFrame(
                    dict(
                        features=self.dp.get_estimator_features_list(),
                        loc_importance=self.estimator.bagging_members[0].feature_importances_[0],
                        scale_importance=self.estimator.bagging_members[0].feature_importances_[1]
                    )
                )

                # Prepare importance rank for loc. Sort by importance, reset index
                # and use new index as rank.
                .sort_values('loc_importance',ascending=False)
                .reset_index(drop=True)
                .reset_index()
                .rename(columns={'index': 'loc_rank'})

                # Prepare importance rank for scale
                .sort_values('scale_importance',ascending=False)
                .reset_index(drop=True)
                .reset_index()
                .rename(columns={'index': 'scale_rank'})

                # Make rank start with 1, not 0
                .assign(
                    loc_rank   = lambda df: df.loc_rank   + 1,
                    scale_rank = lambda df: df.scale_rank + 1,
                )

                # Data wrangling for easy transformation into a dict needed by
                # R3 metrics framework
                .set_index('features')
                .stack()
                .reset_index()

                # Fabricate metric names
                .assign(
                    level_1 = lambda df: df.level_1.str.replace('_',':'),
                    name    = lambda df: mprefix + df.features + ':' + df.level_1
                )
                .rename(columns={0: 'value_number'})
                .set_index('name')
                .sort_index()

                # Get the Series with values only
                .value_number

                # Finaly, export as a Python dict
                .to_dict()
            )



    def median_ape(y_true: pd.Series, y_pred: pd.Series) -> float:
        # Med(|(ŷ-y)÷y|)
        return numpy.median(
                numpy.abs(
                    (y_true - y_pred) / y_true
                )
            )



    def median_pe(y_true: pd.Series, y_pred: pd.Series) -> float:
        # Med(|(y-ŷ)÷ŷ|)
        return numpy.median(
                    (y_pred - y_true) / y_true
                )




    ######################################################################################
    ##
    ## Object save and load in pickle format
    ##
    ######################################################################################



    def _handle_xz(file_obj, mode):
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
            resolved_path=s3path.S3Path.from_uri(resolved_path)
        else:
            resolved_path=pathlib.Path(resolved_path).resolve()

        filename_tpl  = '{dataprovider} • {time} • {full_train_id}.pkl'
        filename_tpl += '.xz' if compress else ''

        filename=filename_tpl.format(
            dataprovider=self.dp.id,
            full_train_id=self.get_full_train_id(),
            time=self.trained_str()
        )

        target = resolved_path / filename

        self.log(
            'Serialized trained model object to {target}'.format(
                target=urllib.parse.unquote(target.as_uri())
            )
        )

        smart_open.register_compressor('.xz', Model._handle_xz)

        # Write to object storage or maybe filesystem
        with smart_open.open(urllib.parse.unquote(target.as_uri()), "wb") as f:
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
            resolved_path=ConfigManager.get('TRAINED_MODELS_PATH', default=path)

        if not resolved_path:
            raise Exception("Don’t know where to load pre-trained models from. Set TRAINED_MODELS_PATH.")

#         self.log(f'Pre-trained model storage is {resolved_path}')

        if resolved_path.startswith('s3://'):
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
        s3path_bug_101=True # https://github.com/liormizr/s3path/issues/101
        if type(resolved_path)==s3path.S3Path and s3path_bug_101:
            # If s3path module still has the bug #101, we'll use s3fs to find
            # matches for the files we need.
            # When bug is resolved, we can use the more generic PathLib method
            # inside `else`.
            import s3fs
            s3=s3fs.S3FileSystem()
            available=[
                resolved_path / pathlib.PurePath(f).relative_to(str(resolved_path)[1:])
                for f in s3.glob(str(resolved_path / filename)[1:]) if '.dvc' not in str(f)
            ]
        else:
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
            if (isinstance(as_of,str) and as_of != '*') or (isinstance(as_of,pd.Timestamp) or isinstance(as_of,datetime.datetime) or isinstance(as_of,datetime.date)):
                # Try to convert string to datetime
                as_of=pd.to_datetime(as_of).to_pydatetime()

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

        smart_open.register_compressor('.xz', Model._handle_xz)

        with smart_open.open(urllib.parse.unquote(model_file.as_uri()), 'rb') as f:
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



    ######################################################################################
    ##
    ## Operational procedures
    ##
    ######################################################################################



    def cleanup(self):
        """
        Delete all internal data used for training and batch predict
        """

        self.log("Flushing internal datasets")

        attr=[
            'sets',
            'batch_predict_data',
            'batch_predict_valuations',
            'batch_predict_time',
        ]

        for a in attr:
            if hasattr(self, a):
                delattr(self,a)



    def __init__(self,
                    dp:                            DataProvider,
                    coach:                         Coach = None,
                    estimator_class                            = Estimator,

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

        - estimator_only: False loads and returns an entire Model object as it was
        pickled, complete with methods and pickled DataProvider object. True, loads only
        the estimator object and some IDs, DataProvider object and methods will not come
        from pickle. Use with caution because loaded estimator might be incompatible with
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
                    raise Exception("This type of Model loading requires a DataProvider object, not just a string. Use DataProviderFactory.get('model_name') to get it.")
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
            # Initial and untrained data passed to __init__
            self.dp                       = dp
            self.trained                  = trained
            self.train_id                 = None
            self.train_session_id         = None
            self.train_queries_signatures = dict()

            # Store more contextual and informative metadata from the Coach and environment
            self.user_name         = pwd.getpwuid(os.getuid())[0]
            self.host_name         = socket.gethostname()
            self.git_branch        = self.coach.git_repo.head.name if self.coach.git_repo else None
            self.git_commit        = self.coach.git_repo.head.target.hex if self.coach.git_repo else None
            self.github_actor      = self.get_config('GITHUB_ACTOR', None)
            self.github_workflow   = self.get_config('GITHUB_WORKFLOW', None)
            self.github_run_id     = self.get_config('GITHUB_RUN_ID', None)
            self.github_run_number = self.get_config('GITHUB_RUN_NUMBER', None)
            
            if estimator_class == Estimator:
                # Got a pure useless Estimator.
                # To not break things, initialize it without parameters from DP
                # because, well, they are useless to it.
                self.estimator     = estimator_class()
            elif self.dp.hollow:
                
                self.estimator     = None
            else:
                self.estimator     = estimator_class(**self.dp.get_estimator_parameters())
                
        if delayed_prereq_binding is False:
            self.load_pre_req_model()



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
            return ConfigManager.get(config_item, default=default, cast=cast)
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

        with self.coach.get_db_connection('xingu').begin() as conn:
            if status == 'train_dataprep_start':
                # train_dataprep_start is the first step of a training session, so
                # register also the overall train
                conn.execute(
                    self.coach.tables['training'].insert().values(
                        train_session_id   = self.train_session_id,
                        train_id           = self.train_id,
                        dataprovider_id    = self.dp.id,
                        user_name          = self.user_name,
                        host_name          = self.host_name,
                        git_branch         = self.git_branch,
                        git_commit         = self.git_commit,
                        github_actor       = self.github_actor,
                        github_workflow    = self.github_workflow,
                        github_run_id      = self.github_run_id,
                        github_run_number  = self.github_run_number,
                        x_features         = json.dumps(self.dp.get_features_list()),
                    )
                )

            conn.execute(
                self.coach.tables['training_status'].insert().values(
                    time               = round(datetime.datetime.utcnow().timestamp()),
                    dataprovider_id    = self.dp.id,
                    train_session_id   = self.train_session_id,
                    train_id           = self.train_id,
                    estimator          = str(self.estimator) if hasattr(self, 'estimator') else None,
                    hyperparam         = json.dumps(self.hyperparam),
                    status             = status
                )
            )



    def fit_and_batch_predict(self):
        self.fit()
        self.batch_predict()



    def get_full_train_id(self):
        if self.train_session_id:
            return f'{self.train_session_id}∶{self.train_id}'
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
        cache_path=self.get_config('QUERY_CACHE_PATH', default=None)
        if cache_path:
            cache_template="cache • {context} • {dp} • {part}"

            for part in self.sets:
                cache_file_name_prefix=cache_template.format(
                    context    = self.context,
                    dp         = self.dp.id,
                    part       = part,
                )

                cache_file_name=cache_file_name_prefix + '.parquet'

                cache_file=pathlib.Path(
                    cache_path,
                    cache_file_name
                ).resolve()

                self.log(f'Saving part on {cache_file}')
                self.sets[part].to_parquet(cache_file, compression='gzip')



    def data_source_to_data(self, sourceid: str, data_source: dict) -> pd.DataFrame:
        """
        data_source is one entry of a complex DataProvider.train_data_sources or
        DataProvider.batch_predict_data_sources

        data_source has structure dict(
            source='units' | 'xingu',
            query="SELECT ..."
        )

        sourceid is the name of this source, as it appears in the DP's dict.

        This method is called in parallel by data_sources_to_data().
        """

        # Initialize to a non-sense value what we are going to return
        df=None

        if data_source is None:
            return df

        cache_path=self.get_config('QUERY_CACHE_PATH', default=None)
        dvc_cache_path=self.get_config('DVC_QUERY_CACHE_PATH', default=None)

        if cache_path is None:
            self.log(
                level=logging.WARNING,
                message="QUERY_CACHE_PATH is not set, which is good for production environment. But it will make you query the DB on every run."
            )

        cache_template="cache • {context} • {dp} • {sourceid} • {signature}"

        # Compute a unique hash for the SQL query text, for cache management purposes
        if cache_path or dvc_cache_path:
            # Compute cache file name using a signature from SQL query text
            cypher=hashlib.shake_256()
            cypher.update(data_source['query'].encode('UTF-8'))
            signature=cypher.hexdigest(10)
            
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

            cache_file=pathlib.Path(
                cache_path,
                cache_file_name
            ).resolve()

            # Check if we have a file with this name
            if os.path.isfile(cache_file):
                # We have a cache hit. Use it.

                self.log(
                    'Using cache from {cache_file} instead of DB for «{source}» on {context}'.format(
                        source      = data_source['source'],
                        cache_file  = cache_file,
                        context     = self.context
                    )
                )

                df=pd.read_parquet(cache_file)
            else:
                # No cache. Retrieve data from DB and make cache.

                self.log(
                    'No cache for «{source}» on {context}, looked for in file {cache_file}. Retrieving data from DB.'.format(
                        source      = data_source['source'],
                        cache_file  = cache_file,
                        context     = self.context
                    )
                )

        if df is None:
            # No success with cache so far.
            # Use DB.

            source_db=self.coach.get_db_connection(data_source['source'])

            self.log(
                level=logging.DEBUG,
                message='Retrieving dataset named «{sourceid}» from «{source}»:\n{query}'.format(
                    sourceid   = sourceid,
                    source     = data_source['source'],
                    query      = textwrap.indent(textwrap.dedent(data_source['query']),'   ')
                )
            )

            # Hit the database
            df=pd.read_sql_query(
                data_source['query'],
                con=source_db
            )

            if cache_path:
                self.log(f'Making cache on {cache_file}')
                df.to_parquet(cache_file, compression='gzip')

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



    def data_sources_to_data(self, data_sources: dict=None) -> dict:
        """
        Get the DataProvider's train_data_sources and batch_predict_data_sources (passed
        as data_sources parameters), which contain a collection of SQL queries and their
        sources, and execute queries in parallel against its specified source DBs.

        Returns a dict with similar structure containing DataFrames with data returned
        by these queries. Ready to be passed to DataProvider.clean_data().
        """

        if data_sources is None:
            return None

        collected_data=dict()

        # Iterate over all data sources, decide wether to use cache or execute query,
        # save query data to cache if QUERY_CACHE_PATH is set.
        with concurrent.futures.ThreadPoolExecutor(thread_name_prefix='data_sources_to_data') as e:
            tasks={
                # Submit all queries or cache data loading in parallel, track them in a
                # dict comprehension.
                e.submit(self.data_source_to_data,sourceid,data_sources[sourceid]): sourceid
                for sourceid in data_sources.keys()
            }

            # Process results of all tasks as they finish
            for task in concurrent.futures.as_completed(tasks):
                collected_data[tasks[task]]=task.result()

        return collected_data



    def trained_str(self):
        time_tpl='%Y.%m.%d-%H.%M.%S'
        return self.trained.strftime(time_tpl)



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
        # Be selective on what to pickle: exclude coach, logger and DataFrames used
        # and generated throughout train and batch predict session.
        
        attributes={
            # DataProvider
            'dp',
            
            # Train info
            'train_session_id',     'train_id',    'trained',
            'hyperopt_strategy',    'hyperparam',  'train_queries_signatures',

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
        Return a dict ready to be converted into YAML for currents.yaml
        """
        attributes={
            # Train info
            'train_session_id',     'train_id',    'trained',
            'hyperopt_strategy',    'hyperparam',  'train_queries_signatures',

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
