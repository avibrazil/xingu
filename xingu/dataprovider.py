import dataclasses
import inspect
import logging
import datetime
import pandas

from .estimator import Estimator

@dataclasses.dataclass
class DataProvider(object):

    ###########################################################
    ###
    ###   Attributes every DataProvider must define
    ###
    ###########################################################

    id:                            str    = dataclasses.field(default=None)
    x_features:                    list   = dataclasses.field(default_factory=list)
    x_estimator_features:          list   = dataclasses.field(default_factory=list)
    y:                             str    = dataclasses.field(default=None)
    train_dataset_sources:         dict   = dataclasses.field(default_factory=dict)
    batch_predict_dataset_sources: dict   = dataclasses.field(default_factory=dict)
    hollow:                        bool   = dataclasses.field(default=False)



    ###########################################################
    ###
    ###   Attributes for data splitting and randomization
    ###
    ###########################################################

    random_state:                  int    = dataclasses.field(default=42)
    test_size:                     float  = dataclasses.field(default=0.1)
    val_size:                      float  = dataclasses.field(default=0.2)

    ###########################################################################
    ###
    ###   Attributes for the Estimator
    ###
    ###   See methods:
    ###     get_estimator_class()
    ###     get_estimator_optimization_search_space()
    ###     get_estimator_parameters()
    ###     get_estimator_hyperparameters()
    ###
    # The (hyper)params is what a SciKit estimator gets in its
    # __init__() method. They are of 2 types:
    #
    #  - estimation quality parameters (those that are tunned and are commonly
    #      called hyper-parameters)
    #  - operational parameters (control operation, verbosity etc)
    #
    # estimator_hyperparams + estimator_params is what is used to initialize
    # the object. Like:
    #     XGBClassifier(**estimator_params,**estimator_hyperparams)
    #
    # We are separating them here because estimator_hyperparams will be
    # optimized while estimator_params are kind of fixed.
    #
    # estimator_hyperparams_search_space and estimator_hyperparams usually
    # contain same keys. The search_space has value ranges and semantics
    # determined by your optimization framework as optuna or skopt.
    #
    # The estimator_class_params is random parameters for the xingu.Estimator
    # class.
    #
    # A xingu.Estimator class will be initialized by xingu.Model like this:
    #
    #    xingu.Estimator(
    #         **DataProvider.estimator_class_params,
    #         params      = DataProvider.estimator_params,
    #         hyperparams = DataProvider.estimator_hyperparams,
    #    )
    #
    ###########################################################################

    estimator_class:                      type   = Estimator
    estimator_class_params:               dict   = dataclasses.field(default_factory=dict)
    estimator_params:                     dict   = dataclasses.field(default_factory=dict)
    estimator_hyperparams:                dict   = dataclasses.field(default_factory=dict)
    estimator_hyperparams_search_space:   dict   = dataclasses.field(default_factory=dict)


    ###########################################################
    ###
    ###   Control batch predict and metrics computers.
    ###
    ###########################################################

    # For classification models, the index of predict_proba result tha matches
    # the training target
    proba_class_index:                    int    = dataclasses.field(default=0)


    ###########################################################
    ###
    ###   Methods to post-process what dataclass left for us
    ###
    ###########################################################

    def __post_init__(self):
        if type(self.estimator_class) == str:
            # Convert an estimator class string into its real class.
            # For example, the string 'xingu.estimators.xgboost_optuna.XinguXGBoostClassifier'
            # will be converted to its real class.
            import importlib

            mod=importlib.import_module('.'.join(self.estimator_class.split('.')[:-1]))
            self.estimator_class=getattr(mod,self.estimator_class.split('.')[-1])

        if issubclass(self.estimator_class, Estimator)==False:
            raise RuntimeError(f"Invalid class: {self.estimator_class}. Must be subclass of xingu.Estimator.")


    ###########################################################
    ###
    ###   Methods that need specialized implementation on
    ###   derived Data Providers.
    ###
    ###########################################################


    def clean_data_for_train(self, datasets: dict) -> pandas.DataFrame:
        """
        datasets is a dict with format:

            {
                'NAME_OF_DATASET1': pandas.DataFrame,
                'NAME_OF_DATASET2': pandas.DataFrame,
                ...
            }

        This method needs to integrate all these DataFrames and return a single
        DataFrame with minimum cleanup.

        Provides a default implementation that simply concatenates all
        datasets. This is probably very wrong for your DataProvider, so please
        reaimplement.
        """
        result = None

        for d in datasets:
            if result is None:
                result = datasets[d]
            else:
                result = pandas.concat(
                    [
                        result,
                        datasets[d]
                    ]
                )

        return result



    def feature_engineering_for_train(self, data: pandas.DataFrame) -> pandas.DataFrame:
        """
        Applies any post-cleanup feature engineering on the DataFrame returned
        by clean_data().

        This must be implemented on a derived class.
        """
        return data



    def last_pre_process_for_train(self, data: pandas.DataFrame) -> pandas.DataFrame:
        """
        Last chance to process data returned by train_feature_engineering() and before training.

        This must be implemented on a derived class.
        """
        return data



    def clean_data_for_batch_predict(self, datasets: dict) -> pandas.DataFrame:
        # Default implementation; please reimplement
        return self.clean_data_for_train(datasets)



    def feature_engineering_for_batch_predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        return data



    def last_pre_process_for_batch_predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        """
        Last chance to process data returned by feature_engineering_for_batch_predict() and before batch prediction.

        This must be implemented on a derived class.
        """
        return data



    # Implement this if you need to fabricate features for your estimator. And in this
    # case, x_estimator_features and x_features might probably be different
    def feature_engineering_for_predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        """
        User must call this before all pred_*() methods in order to prepare data for the estimator.

        Here might rest code shared by feature_engineering_for_train() and feature_engineering_for_batch_predict()
        """
        return data



    def get_estimator_class(self) -> type:
        return self.estimator_class



    def get_estimator_class_params(self) -> dict:
        return self.estimator_class_params



    def get_estimator_params(self) -> dict:
        return self.estimator_params



    def get_estimator_hyperparams(self) -> dict:
        return self.estimator_hyperparams



    def get_estimator_optimization_search_space(self) -> dict:
        return self.estimator_hyperparams_search_space



    def post_process_after_train(self, model):
        """
        Called after estimator was trained but before PKL is saved or metrics computed.
        Implement it in your DataProvider if you need to compute coefficients or
        manipulate your Model object somehow.

        If you’ll compute things that will be used in future predicts, don’t forget to
        save them in the PKL implementing your own __getstate__() method.

        The `model` parameter has the entire Model object and the `dp` attribute
        inside `model` equals to `self`. In other words, `model.dp` and `self` are the
        same object in the context of this method. Yet in another set of
        words, `model.dp == self`.
        """
        pass



    def pre_process_for_generic_predict(self, X: pandas.DataFrame, model) -> pandas.DataFrame:
        """
        Virtual method to be implemented in concrete DPs in case your pre-processing is
        the same for predict() and predict_proba().

        If pre-processing is not the same for these methods, implement both
        pre_process_for_predict() and pre_process_for_predict_proba()
        """
        return X



    def post_process_after_generic_predict(self, X: pandas.DataFrame, Y_pred: pandas.DataFrame, model) -> pandas.DataFrame:
        """
        Virtual method to be implemented in concrete DPs in case your post-processing is
        the same for predict() and predict_proba().

        If post-processing is not the same for these methods, implement both
        post_process_after_predict() and post_process_after_predict_proba()
        """
        return Y_pred



    def pre_process_for_predict(self, X: pandas.DataFrame, model) -> pandas.DataFrame:
        """
        Called by Model.generic_predict() right before X is passed to its internal
        estimator to compute Ŷ.

        Your implementation may modify X completely, and what you return here
        will be passed to Model's internal estimator.

        The abstract implementation here does nothing.
        """
        return self.pre_process_for_generic_predict(X,model)



    def post_process_after_predict(self, X: pandas.DataFrame, Y_pred: pandas.DataFrame, model) -> pandas.DataFrame:
        """
        Called by Model.pred_dist() right after Ŷ (Y_pred) is computed.

        X is whatever you returned in pre_process_for_pred_dist().

        Your implementation may modify Y_pred completely, and what you return here
        will be returned to Model caller.

        The abstract implementation here does nothing.
        """
        return self.post_process_after_generic_predict(X,Y_pred,model)



    def pre_process_for_predict_proba(self, X: pandas.DataFrame, model) -> pandas.DataFrame:
        """
        Called by Model.generic_predict() right before X is passed to its internal
        estimator to compute Ŷ.

        Your implementation may modify X completely, and what you return here
        will be passed to Model's internal estimator.

        The abstract implementation here does nothing.
        """
        return self.pre_process_for_generic_predict(X,model)



    def post_process_after_predict_proba(self, X: pandas.DataFrame, Y_pred: pandas.DataFrame, model) -> pandas.DataFrame:
        """
        Called by Model.pred_dist() right after Ŷ (Y_pred) is computed.

        X is whatever you returned in pre_process_for_pred_dist().

        Your implementation may modify Y_pred completely, and what you return here
        will be returned to Model caller.

        The abstract implementation here does nothing.
        """
        return self.post_process_after_generic_predict(X,Y_pred,model)





    ###########################################################
    ###
    ###   Generic methods used as is by all derived classes.
    ###   But derived classes can reimplement a specialization
    ###   of them.
    ###
    ###########################################################


    def get_logger(self):
        if not hasattr(self,'logger'):
            self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        return self.logger



    def log(self, message='', level=logging.INFO):
        self.get_logger().log(
            level,
            'id={id}: {message}'.format(
                id=self.id,
                message=message
            )
        )



    def get_dataset_sources_for_train(self, model=None) -> dict:
        """
        Returns the train_dataset_sources, which is a dicts with format:

            {
                'NAME_OF_DATASET1': {
                    'source': 'units',
                    'query': 'select ...'
                },
                'NAME_OF_DATASET2': {
                    'source': 'xingu',
                    'query': 'select ...'
                },
                'NAME_OF_DATASET3': {
                    'source': 'units',
                    'query': 'select ...'
                }
            }
        """
        return self.train_dataset_sources



    def get_dataset_sources_for_batch_predict(self, model=None) -> dict:
        return self.batch_predict_dataset_sources



    def get_features_list(self) -> list:
        return self.x_features



    def get_estimator_features_list(self) -> list:
        if hasattr(self,'x_estimator_features') and len(self.x_estimator_features)>0:
            # If we actually have x_estimator_features...
            return self.x_estimator_features
        else:
            return self.x_features



    def get_target(self) -> str:
        return self.y



    def __repr__(self) -> str:
        return '{klass}(id={id})'.format(id=self.id, klass=type(self).__name__)



    ###########################################################
    ###
    ###   Generic data split methods
    ###
    ###########################################################


    def random_data_split(data: pandas.DataFrame, test_size: float, val_size: float) -> (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame):
        """
        Split data in 3 parts as:

        - train (size = total - (test_size + val_size))
        - val (size = val_size)
        - test (size = test_size)

        Data are split with sklearn.model_selection.train_test_split().

        test_size and val_size must be float, between 0.0 and 1.0 and represent the
        proportion of split.

        Derived classes might have more specialized implementations to split data.
        """

        from sklearn.model_selection import train_test_split

        df_train, df_test = train_test_split(
            data,
            test_size=test_size,
            random_state=self.random_state
        )

        df_train, df_val = train_test_split(
            df_train,
            test_size=val_size,
            random_state=self.random_state
        )

        return (df_train, df_val, df_test)



    def time_data_split(df, date_column, delta_time_days):
        """
        Split data in 2 parts as:

        - train (all data where date_column is prior than delta_time_days ago)
        - test (all data where date_column is after delta_time_days ago)

        In summary, test will have the latest delta_time_days data and train will have all
        the rest, which is older.

        delta_time_days is an integer number of days.

        date_column is the name of date column in the df dataset.
        """
        split_date = datetime.datetime.now() - datetime.timedelta(delta_time_days)

        df_train = df[df[date_column] < split_date.date()]
        df_test = df[df[date_column] >= split_date.date()]

        return df_train, df_test



    def data_split_for_train(self, data: pandas.DataFrame) -> dict:
        """
        Split data before train.
        Implement in your DataProvider.

        Must return a dict of dataframes. The key named 'train' will be used
        for training. Other dataframes can be used for metrics computation or
        other purposes in your DataFrame.
        """
        return dict(train=data)


    ####################################################################
    ###
    ###   Hooks for customized and extensible metrics computation
    ###
    ####################################################################


    def get_metrics_computers(self, type: str='estimation') -> list:
        """
        Return a list of member functions that can compute metrics for estimator outputs.

        Current types that will be scanned are:

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
        Return a list of member functions that can compute metrics for estimator outputs.

        Current types that will be scanned are:

        • trainsets_model
        • batch_model
        • global_model
        """


        methods=[]

        for method in inspect.getmembers(self, predicate=inspect.ismethod):
            if 'render_' + type + '_plots_' in method[0]:
                methods.append(method[1])

        return methods



    def __getstate__(self):
        return dict(
            x_features                         = self.x_features,
            x_estimator_features               = self.x_estimator_features,
            y                                  = self.y,
            estimator_class                    = self.estimator_class,
            estimator_class_params             = self.estimator_class_params,
            estimator_params                   = self.estimator_params,
            estimator_hyperparams              = self.estimator_hyperparams,
            estimator_hyperparams_search_space = self.estimator_hyperparams_search_space,
            proba_class_index                  = self.proba_class_index,
            train_dataset_sources              = self.train_dataset_sources,
            batch_predict_dataset_sources      = self.batch_predict_dataset_sources,
            random_state                       = self.random_state,
            test_size                          = self.test_size,
            val_size                           = self.val_size,
            hollow                             = self.hollow,
        )
