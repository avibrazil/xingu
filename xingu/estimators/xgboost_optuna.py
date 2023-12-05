import os
import logging
import concurrent.futures
import decouple
import numpy
import pandas
import xgboost
import optuna

import xingu


class XinguXGBoostClassifier(xingu.Estimator):
    """
    Multi-XGBoost implementation of an estimator optimized by Optuna.
    """


    def __init__(self, params: dict=None, hyperparams: dict=None, random_state=42, bagging_size=1, optimization_trials=10, **kwargs):
        super().__init__(params=params,hyperparams=hyperparams)

        self.bagging_size=bagging_size
        self.optimization_trials=optimization_trials
        self.bagging_members=[]

        self.random_state=random_state



    def __repr__(self):
        template='{klass}(size={size}, random_state={random_state}, members={members})'

        text = template.format(
            klass          = type(self).__name__,
            size           = self.bagging_size,
            random_state   = self.random_state,
            members        = self.bagging_members
        )

        # Remove extra spaces and \n
        text = ' '.join(text.split())

        return text



    def hyperparam_optimize(self, model):
        import sklearn

        def objective(trial, model):
            """
            Your DataProvider must have something like:

                estimator_hyperparam_search_space = dict(
                    iterations              = ('int',        dict(low=10,    high=500)),
                    depth                   = ('int',        dict(low=1,     high=9)),
                    border_count            = ('int',        dict(low=1,     high=255)),
                    l2_leaf_reg             = ('int',        dict(low=2,     high=35)),
                    learning_rate           = ('loguniform', dict(low=0.01,  high=1.0)),
                    random_strength         = ('loguniform', dict(low=1e-9,  high=10)),
                    bagging_temperature     = ('float',      dict(low=0.0,   high=1.0)),
                    scale_pos_weight        = ('uniform',    dict(low=0.01,  high=1.0))
                )

            This will be converted and used to optimize an estimator.
            """
            datasets     = model.sets
            features     = model.dp.get_estimator_features_list()
            target       = model.dp.get_target()
            search_space = model.dp.get_estimator_optimization_search_space()

            suggested_hyperparams = {
                # Convert the DataProvider.estimator_hyperparam_search_space
                # dict into Optuna´s trial.suggest_*() calls
                p: getattr(trial,'suggest_' + search_space[p][0])(
                    p,
                    **search_space[p][1]
                )
                for p in search_space
            }

            skf = sklearn.model_selection.StratifiedKFold(
                n_splits=self.bagging_size,
                shuffle=True,
                random_state=self.random_state,
            )

            predicts=pandas.DataFrame(index=datasets['train'].index).assign(proba=None)
            for (itrain,ival) in skf.split(
                        datasets['train'],
                        datasets['train'].stratify
                    ):

                # self.log('All params: ' + str(dict(
                #     random_state=self.random_state,
                #     **self.params,
                #     **suggested_hyperparams
                # )))

                classifier = xgboost.XGBClassifier(
                    random_state=self.random_state,
                    **self.params,
                    **suggested_hyperparams
                )

                # self.log(datasets['train'].iloc[itrain].head(10).to_markdown())

                classifier.fit(
                    X=datasets['train'].iloc[itrain][features],
                    y=datasets['train'].iloc[itrain][target],
                    eval_set=[
                        (datasets['train'].iloc[itrain][features], datasets['train'].iloc[itrain][target]),
                        (datasets['train'].iloc[ival][features],   datasets['train'].iloc[ival][target]),
                    ]
                )

                # Cirurgically set predicts in current ival rows
                predicts.proba.iloc[ival] = classifier.predict_proba(
                    datasets['train'].iloc[ival][features]
                )[:, model.dp.proba_class_index]


            return sklearn.metrics.roc_auc_score(
                datasets['train'][target],
                predicts.proba
            )


        optimizer=optuna.create_study(
            study_name='Xingu generic XGBoostClassifier optimizer',
            direction='maximize'
        )

        optimizer.optimize(lambda trial: objective(trial, model), n_trials=self.optimization_trials)

        # for i in range(20):
        #     ntrials = len(optimizer.trials)
        #     optimizer.optimize(objective, n_trials=min(500-ntrials,50))

        # Convert the OrderedDict returned by these objects into a plain dict
        # return {i[0]:i[1] for i in optimizer.best_params_.items()}
        return optimizer.best_trial.params



    def fit_single(self, data, itrain, ival, features, target) -> xgboost.XGBClassifier:
        import sklearn

        clf = xgboost.XGBClassifier(
            random_state=self.random_state,
            **self.params,
            **self.hyperparams
        )

        # Actual training session begins

        clf.fit(
            X=data.iloc[itrain][features],
            y=data.iloc[itrain][target],
            eval_set=[
                (data.iloc[itrain][features], data.iloc[itrain][target]),
                (data.iloc[ival][features],   data.iloc[ival][target]),
            ]
        )

        return clf



    def fit(self, datasets, features, target):
        import sklearn
        # Add attribute 'max_workers=1' to inhibit parallelism

        max_workers=decouple.config('PARALLEL_ESTIMATORS_MAX_WORKERS', default=0, cast=int)
        if max_workers == '' or max_workers == 0:
            max_workers=None
            self.logger.info(f'Unlimited parallel estimators to train')
        else:
            self.logger.info(f'{max_workers} parallel estimators to train')

        skf = sklearn.model_selection.StratifiedKFold(
            n_splits=self.bagging_size,
            shuffle=True,
            random_state=self.random_state,
        )

        executor=concurrent.futures.ThreadPoolExecutor(thread_name_prefix='fit', max_workers=max_workers)
        tasks=[]
        index=0
        for (itrain,ival) in skf.split(
                    datasets['train'],
                    datasets['train'].stratify
                ):
            
            self.log(f'Trigger parallel train of XGBoost estimator #{index+1} of {self.bagging_size}...')

            tasks.append(
                executor.submit(
                    # Method name to call asynchronously
                    self.fit_single,
                    # Its parameters
                    datasets['train'],
                    itrain,
                    ival,
                    features,
                    target,
                )
            )
            
            index += 1

        self.log(f'Waiting for all member training to finish in parallel')

        # Process result as soon as it is available
        index=0
        for task in concurrent.futures.as_completed(tasks):
            e=task.exception()
            if e is None:
                # Success
                self.bagging_members.append(task.result())
                index+=1
                self.log(f'Finished train of member #{index} of {self.bagging_size}...')
            else:
                # Failure
                raise e



    def predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        return self.generic_predict(data)



    def predict_proba(self, data: pandas.DataFrame, class_index: int=None) -> pandas.DataFrame:
        return self.generic_predict(
            data,
            method='predict_proba',
            class_index=class_index
        )



    def generic_predict(self, data: pandas.DataFrame, method='predict', class_index: int=None) -> pandas.DataFrame:
        """
        Predict using predict_multi() and then consolidate and aggregate estimations
        for all bagging members.
        """

        duplicate_index_error_message=[
            'DataFrame index has duplicates, which will lead',
            'to sever inconsistencies in multi-model sumarization.',
            'Here is a sample of problematic IDs; clean',
            'DataFrame appropriatelly: {}'
        ]

        duplicate_index_error_message=' '.join(duplicate_index_error_message)

        if data.shape[0]<1:
            # Empty input. Return an empty standard output.
            return pandas.DataFrame(columns=['estimation'])

        index_duplicate_count=data.index.value_counts()
        if index_duplicate_count.iloc[0] > 1:
            raise OverflowError(
                duplicate_index_error_message
                .format(list(index_duplicate_count[index_duplicate_count>1].head(10).index))
            )

        def ddebug(table,message):
            self.log(message,level=logging.DEBUG)
            return table


        if self.bagging_size>1:
            # Define aggregation strategy and compute aggregations for multiple estimations
            # in an ensamble/bagging environment.

            # Default aggregation function is mean().
            # But if this is a classifier, predict() returns a class, so we need to find the mode().
            # Aggregation strategy applies the correct aggregation function based if predict() or predict_proba()
            agg_function='mean'
            if self.is_classifier() and method == 'predict':
                agg_function=pandas.Series.mode

            agg_strategy=dict(estimation=agg_function)
            if method == 'predict_proba' and class_index is None:
                agg_strategy={
                    f'estimation_class_{col}':agg_function
                    for col in self.bagging_members[0].classes_
                }

            self.log(f"Bagging aggregation strategy: {agg_strategy}",level=logging.DEBUG)

            return (
                self.predict_multi(data,method,class_index)

                # Group by index
                .groupby(level=0)

                # Member ID was consolidated and ghosted at this point.

                # Aggregated results by mean or mode (only if classifier and predict method).
                # One of the reasons Index can’t have duplicates is to avoid
                # bias in here.
                .agg(agg_strategy)

                # Order DataFrame the same as input
                .reindex(data.index)
            )
        else:
            return self.predict_single(data,method,class_index).drop(columns=['member'])



    ####################################################################################
    ###
    ### Internal methods
    ###
    ####################################################################################



    def predict_single(self, data: pandas.DataFrame, method='predict', class_index: int=None, bagging_member: int=0) -> pandas.DataFrame:
        """
        Return raw estimations of one estimator member tagged by member ID.
        If method is 'predict_proba' all class probabilities are returned in multiple
        columns called estimation_class_{I}. Unless class_index is passed, then only
        that class is returned.
        
        Resulting dataframe has columns:

        - index
        - estimator member ID (0,1,2,3 etc as passed in bagging_member)
        - estimation (if predict) or estimation_class_{I} (if predict_proba)

        This method was designed to be parallelized with concurrent.futures.
        """
        
        self.log(f'Member #{bagging_member} is predicting for {data.shape[0]} datapoints...',level=logging.DEBUG)
        
        def ddebug(table,message):
            self.log(message,level=logging.DEBUG)
            return table
        
        renamer=lambda col: 'estimation'
        unwanted_classes=list()
        
        if method == 'predict_proba':
            if class_index is not None:
                # Want single value, not probabilities of all classes
                unwanted_classes=list(set(self.bagging_members[bagging_member].classes_)-set([class_index]))
            else:
                # Change strategy for column renamer
                renamer=lambda col: f'estimation_class_{col}'
        
        return (
            pandas.DataFrame(
                index=data.index,
                
                # Compute estimation via predict or predict_proba, using 1 member
                data=getattr(self.bagging_members[bagging_member], method)(data)
            )
            
            # Keep only the class_index class, if provided
            .drop(columns=unwanted_classes)
            
            # Rename columns from 0, 1, etc to estimation (if predict) or estimation_class_{I}
            .rename(columns=renamer)
            
            # Tag it with member ID
            .assign(member=bagging_member)
            
            # Reduce RAM usage
            .assign(
                member=lambda table: table.member.astype('category')
            )
            
            .pipe(lambda table: ddebug(table,f'Member #{bagging_member} finished predicting for {data.shape[0]} datapoints.'))
        )



    def predict_multi(self, data: pandas.DataFrame, method='predict', class_index: int=None) -> pandas.DataFrame:
        """
        Return raw estimations of all estimator members tagged by member ID in
        a MultiIndex DataFrame.

        Resulting dataframe has columns:

        Index:
        - (original index, preferably unit_id)
        - estimator member ID (0,1,2,3 etc)

        Columns:
        - estimation (if predict) or estimation_class_{I} (if predict_proba)
        """

        executor=concurrent.futures.ThreadPoolExecutor(thread_name_prefix='pred_multi_dist')
        estimation=None
        tasks=[]

        # Trigger all estimators in parallel
        for i in range(self.bagging_size):
            tasks.append(
                # Execute in the background ...
                executor.submit(
                    # ... this method ...
                    self.predict_single,

                    # ... first parameter ...
                    data,

                    # ... second parameter ...
                    method,

                    # ... third parameter ...
                    class_index,

                    # ... fourth parameter ...
                    i,
                )
            )

        # Process their resulting DataFrame as soon as it is available
        for task in concurrent.futures.as_completed(tasks):
            if estimation is None:
                estimation=task.result()
            else:
                estimation=pandas.concat([estimation,task.result()])

        return estimation.set_index([estimation.index,'member']).sort_index()



    def __getstate__(self):
        return dict(
            **xingu.Estimator.__getstate__(self),

            # Number of trained models. Should be same as len(bagging_members)
            bagging_size     = self.bagging_size,

            # Array of trained sklearn.tree.DecisionTreeRegressor()s
            bagging_members  = self.bagging_members,

            # Random number used by the class, as 42
            random_state     = self.random_state
        )



    def is_classifier(self):
        from sklearn.base import is_classifier
        return is_classifier(self.bagging_members[0])