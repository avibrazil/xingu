import inspect
from typing import Tuple
import logging
import datetime
import pandas as pd

from typing import Tuple



class DataProvider(object):

    ###########################################################
    ###
    ###   Attributes every DataProvider must define
    ###
    ###########################################################

    # id:                          str    = (must be defined in derived classes)
    x_features:                    list   = []
    x_estimator_features:          list   = []
    y:                             str    = None
    train_dataset_sources:         dict   = None
    batch_predict_dataset_sources: dict   = None
    hollow:                        bool   = False



    ###########################################################
    ###
    ###   Attributes for data splitting and randomization
    ###
    ###########################################################

    random_state:                  int    = 42
    test_size:                     float  = 0.1
    val_size:                      float  = 0.2
    train_split_days:              int    = 150
    date_column_name:              str    = 'date'

    estimator_bagging_size:        int    = 20
    estimator_hyperparams:         dict   = None
    
    ###########################################################
    ###
    ###   Attributes for the purpose of collecting metrics
    ###
    ###########################################################
    
    QUANTILES = [
        ('p_05',0.05),
        ('p_10',0.1),
        ('p_20',0.2),
        ('p_30',0.3),
        ('p_40',0.4),
        ('p_50',0.5),
        ('p_60',0.6),
        ('p_70',0.7),
        ('p_80',0.8),
        ('p_90',0.9),
        ('p_95',0.95)
    ]

    LOWER_BOUND     = "p_20"
    UPPER_BOUND     = "p_80"
    POINT_ESTIMATE  = "p_50"
    
    ###########################################################
    ###
    ###   Methods that need specialized implementation on
    ###   derived Data Providers.
    ###
    ###########################################################


    def clean_data_for_train(self, datasets: dict) -> pd.DataFrame:
        """
        datasets is a dict with format:

            {
                'NAME_OF_DATASET1': pd.DataFrame,
                'NAME_OF_DATASET2': pd.DataFrame
            }

        This method needs to integrate all these DataFrames and return a single
        DataFrame already cleaned up.

        This must be implemented on a derived class.
        """
        return pd.DataFrame()



    def feature_engineering_for_train(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies any post-cleanup feature engineering on the DataFrame returned
        by clean_data().

        This must be implemented on a derived class.
        """
        return data



    def last_pre_process_for_train(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Last chance to process data returned by train_feature_engineering() and before training.

        This must be implemented on a derived class.
        """
        return data



    def clean_data_for_batch_predict(self, datasets: dict) -> pd.DataFrame:
        return pd.DataFrame()



    def feature_engineering_for_batch_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        return data



    def last_pre_process_for_batch_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Last chance to process data returned by feature_engineering_for_batch_predict() and before batch prediction.

        This must be implemented on a derived class.
        """
        return data



    # Implement this if you need to fabricate features for your estimator. And in this
    # case, x_estimator_features and x_features might probably be different
    def feature_engineering_for_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        User must call this before all pred_*() methods in order to prepare data for the estimator.

        Here might rest code shared by feature_engineering_for_train() and feature_engineering_for_batch_predict()
        """
        return data



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



    def pre_process_for_pred_dist(self, X: pd.DataFrame, model) -> pd.DataFrame:
        """
        Called by Model.pred_dist() right before X is passed to its internal
        estimator to compute Ŷ.
        
        Your implementation may modify X completely, and what you return here
        will be passed to Model's internal estimator.
        
        The abstract implementation here does nothing.
        """
        return X
    
    
    
    def post_process_after_pred_dist(self, X: pd.DataFrame, Y_pred: pd.DataFrame, model) -> pd.DataFrame:
        """
        Called by Model.pred_dist() right after Ŷ (Y_pred) is computed.
        
        X is whatever you returned in pre_process_for_pred_dist().
        
        Your implementation may modify Y_pred completely, and what you return here
        will be returned to Model caller.
        
        The abstract implementation here does nothing.
        """
        return Y_pred
    
    
    
    def pre_process_for_pred_quantiles(self, X: pd.DataFrame, model) -> pd.DataFrame:
        """
        Called by Model.pred_dist() right before X is passed to its internal
        estimator to compute Ŷ.
        
        Your implementation may modify X completely, and what you return here
        will be passed to Model's internal estimator.
        
        The abstract implementation here does nothing.
        """
        return X
    
    
    
    def post_process_after_pred_quantiles(self, X: pd.DataFrame, Y_pred: pd.DataFrame, model) -> pd.DataFrame:
        """
        Called by Model.pred_dist() right after Ŷ (Y_pred) is computed.
        
        X is whatever you returned in pre_process_for_pred_dist().
        
        Your implementation may modify Y_pred completely, and what you return here
        will be returned to Model‘s caller.
        
        The abstract implementation here does nothing.
        """
        return Y_pred

    
    
    ###########################################################
    ###
    ###   Generic methods used as is by all derived classes.
    ###   But derived classes can reimplement a specialization
    ###   of them.
    ###
    ###########################################################



    def __init__(self):
        pass



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



    def get_dataset_sources_for_train(self) -> dict:
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



    def get_dataset_sources_for_batch_predict(self) -> dict:
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



    def get_estimator_parameters(self) -> dict:
        # Method optimized for Loft usage of NGBoost.
        # Must return a dict with parameters expected by NGBClassic::__init__()

        return dict(
            bagging_size=self.estimator_bagging_size
        )



    def get_estimator_hyperparameters(self) -> dict:
        if hasattr(self,'estimator_hyperparams'):
            return self.estimator_hyperparams
        else:
            return None



    def __repr__(self) -> str:
        return '{klass}(id={id})'.format(id=self.id, klass=type(self).__name__)



    ###########################################################
    ###
    ###   Generic data split methods
    ###
    ###########################################################


    def random_data_split(data: pd.DataFrame, test_size: float, val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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



    def data_split_for_train(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Convenience default implementation that will be called by Model.fit() method.
        Your DataProvider may implement something different.

        Splits data in 3 parts:

        - train: all data prior self.train_split_days from now
        - val: self.val_size proportion extracted from train data
        - test: data which self.date_column is in the last self.train_split_days

        If self.train_split_days is 150, test will contain last 150 days of data, train
        will contain all the rest except val data.

        """
        from sklearn.model_selection import train_test_split

        train, test = DataProvider.time_data_split(
            data,
            date_column       = self.date_column_name,
            delta_time_days   = self.train_split_days
        )

        train, val = train_test_split(
            train,
            test_size         = self.val_size,
            random_state      = self.random_state
        )

        return (train, val, test)



    ####################################################################
    ###
    ###   Hooks for customized and extensible metrics computation
    ###
    ####################################################################


    def get_metrics_computers(self, type: str='valuation') -> list:
        """
        Return a list of member functions that can compute metrics for estimator outputs.

        Current types that will be scanned are:

        • trainsets_model
        • global_model
        • valuation
        """


        methods=[]

        for method in inspect.getmembers(self, predicate=inspect.ismethod):
            if 'compute_' + type + '_metrics_' in method[0]:
                methods.append(method[1])

        return methods



    def __getstate__(self):
        return dict(
            x_features                      = self.x_features,
            x_estimator_features            = self.x_estimator_features,
            y                               = self.y,
            train_dataset_sources           = self.train_dataset_sources,
            batch_predict_dataset_sources   = self.batch_predict_dataset_sources,
            random_state                    = self.random_state,
            test_size                       = self.test_size,
            val_size                        = self.val_size,
            train_split_days                = self.train_split_days,
            estimator_bagging_size          = self.estimator_bagging_size,
            date_column_name                = self.date_column_name,
            hollow                          = self.hollow
        )
