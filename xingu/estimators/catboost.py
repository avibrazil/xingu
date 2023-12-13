import os
import logging
import concurrent.futures
import decouple
import numpy
import pandas
import catboost

import xingu


class XinguCatBoostClassifier(xingu.Estimator):
    """
    Multi-CatBoost implementation of an estimator suited for Pan operations.
    """


    hyperparam = dict()



    def __init__(self, bagging_size=1, hyperparams: dict=hyperparam, random_state=42):
        super().__init__(hyperparams)

        self.bagging_size=bagging_size
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



    def hyperparam_optimize(self, datasets: dict, features: list, target: str, search_space: dict):
        from sklearn.model_selection import KFold
        from sklearn.metrics import roc_auc_score, make_scorer
        from skopt import BayesSearchCV

        optimizer = BayesSearchCV(
            # Semantics
            estimator            = catboost.CatBoostClassifier(**self.hyperparam),
            cv                   = KFold(n_splits=5, shuffle=True, random_state=2022),
            search_spaces        = search_space,
            scoring              = make_scorer(
                roc_auc_score,
                needs_threshold     = True
            ),

            # Verbosity
            verbose              = 1,

            # Operational
            n_iter               = 200,
            # use just 1 job with CatBoost in order to avoid segmentation fault
            # n_jobs               = 1,
            return_train_score   = False,
            refit                = True,
            optimizer_kwargs     = dict(base_estimator='GP'),
            random_state         = self.random_state,
        )

        optimizer.fit(
            X = datasets['train'][features],
            y = datasets['train'][target],

            # callbacks = [skopt.callbacks.VerboseCallback(100)]
        )

        # Convert the OrderedDict returned by these objects into a plain dict
        return {i[0]:i[1] for i in optimizer.best_params_.items()}



    def fit_single(self, datasets, features, target, index) -> catboost.CatBoostClassifier:
        train_bootstrap = datasets['train'].sample(
            frac               = 1,
            replace            = True,
            random_state       = self.random_state+index
        )

        clf = catboost.CatBoostClassifier(
            **self.hyperparam
        )

        # Actual training session begins
        self.log(f'Trigger parallel train of CatBoost estimator #{index+1} of {self.bagging_size}...')

        clf.fit(
            train_bootstrap[features],
            train_bootstrap[target],
        )

        return clf



    def fit(self, datasets, features, target, model=None):
        # Add attribute 'max_workers=1' to inhibit parallelism

        max_workers=decouple.config('PARALLEL_ESTIMATORS_MAX_WORKERS', default=0, cast=int)
        if max_workers == '' or max_workers == 0:
            max_workers=None
            self.logger.info(f'Unlimited parallel estimators to train')
        else:
            self.logger.info(f'{max_workers} parallel estimators to train')

        executor=concurrent.futures.ThreadPoolExecutor(thread_name_prefix='fit', max_workers=max_workers)
        tasks=[]
        for i in range(self.bagging_size):
            tasks.append(
                executor.submit(
                    # Method name to call asynchronously
                    self.fit_single,
                    # Its parameters
                    datasets,
                    features,
                    target,
                    i
                )
            )

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