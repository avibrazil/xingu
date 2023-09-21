import pandas
import xingu
import xingu.estimators.catboost


class DPMyModel(xingu.DataProvider):
    # The ID of your model is how it will be referenced everywhere.
    # ID-less DataProviders will be ignored by Xingu´s DataProviderFactory (an
    # internal class); they can be used as base classes to other concrete
    # DataProviders.
    #
    # id='my_model'



    # A list of features as seen by the external world
    x_features = """
        feature_1
        feature_2
        feature_3
    """.split()



    # The list of features that will actually train your internal Estimator.
    # Your DataProvider must provide ways to convert x_features into
    # x_estimator_features, and this is done by one of your feature_engineering
    # methods below.
    x_estimator_features = """
        feature_1
        feature_2
        engineered_feature_4
    """.split()



    # Your target variable/column
    y = 'churned'



    # For classifiers, the class probability that should be considered when 
    # predict_proba()
    proba_class_index = 1

    
    
    # Interfaces and definitions for the Estimator object
    
    # The concrete class that should be used as a xingu.Estimator.
    # You probably want to implement this class to put your hyper-parameters
    # optimization code and other details.
    estimator_class = xingu.estimators.catboost.XinguCatBoostClassifier



    # The (hyper)params is what a SciKit estimator gets in its
    # __init__() method. They are of 2 types:
    #
    #  - estimation quality parameters (those that are tunned and are commonly
    #    called hyper-parameters)
    #  - operational parameters (control operation, verbosity etc)
    #
    # estimator_hyperparams + estimator_params is what is used to initialize
    # the object. We are separating them here because estimator_hyperparams
    # will be optimized while estimator_params are kind of fixed.
    #    
    estimator_hyperparams = dict(
        iterations              = 805,
        depth                   = 6,
        border_count            = 45,
        l2_leaf_reg             = 19,
        learning_rate           = 0.01552065618292981,
        random_strength         = 0.0361414296804267,
        bagging_temperature     = 0.8123959883573634,
        scale_pos_weight        = 0.8739040497768427,
    )

    estimator_params = dict(
        verbose                 = False,
        loss_function           = 'Logloss',
        eval_metric             = "AUC",
        od_type                 = 'Iter',
        od_wait                 = 200
    )
    
    
    
    # A dict of SQL queries and/or static URLs. Xingu will retrieve all this
    # data for you and pass to self.clean_data_for_train() as a dict of
    # DataFrames.
    #
    # This attribute won´t be accessed directly. Instead, your
    # get_dataset_sources_for_train() will be called to return a similar dict.
    # This is your chance to generate the data sources dict dynamically. If
    # you do not implement the get_dataset_sources_for_train() method,
    # DataProvider´s implementation will simply return this dict.
    #
    # The names "my_athena", "my_databricks" that appear in the source attribute
    # are nicknames defined by the DATABASES environment variable as:
    #
    # DATABASES="my_athena|awsathena+rest://athena.us-east-1.amaz...|my_pg|postgresql+psycopg2://pg_user:pg_pass@pg-host.com/pg_db"
    # 
    # These DB connections can also be passed to xingu command multiple times as:
    #
    # $ xingu \
    #       --database my_athena awsathena+rest://athena.us-east-1.amaz... \
    #       --database my_pg postgresql+psycopg2://pg_user:pg_pass@pg-host.com/pg_db
    #
    # All datasources defined here will be saved as cache parquets under folder
    # $DATASOURCE_CACHE_PATH. Simply delete the cache files to invalidate cache
    # and force data retrieval again.
    #
    train_dataset_sources = dict(
        my_train_data=dict(
            source='my_athena',
            query="select * from ...",
        ),
        more_train_data=dict(
            source='my_databricks',
            query="select * from ..."
        ),
        open_dataset=dict(
            url="https://storage.googleapis.com/kagglesdsdata/competitions/5407/868283/train.csv",
            params=dict(
                sep='|',
                nrows=1000,
                pandas_read_csv_param3='...',
            )
        ),
        world_cities=dict(
            url="s3://my_bucket/cities_of_the_world.parquet",
            params=dict(
                pandas_read_parquet_param1='...',
                pandas_read_parquet_param2='...',
            )
        )
    )
    
    
    # This follows same idea as train_dataset_sources attribute above but is
    # related to get_dataset_sources_for_batch_predict() method
    batch_predict_dataset_sources = dict(
        my_batch_data=dict(
            source='my_postgresql',
            query="select * from ...",
        ),
    )

    
    
    # Implement this method if you need to pass Xingu a data source dict more
    # dynamic than train_dataset_sources attribute
    # def get_dataset_sources_for_train(model: xingu.Model) -> dict:
    #     return self.train_dataset_sources



    # Implement this method to integrate all datasets requested by
    # get_dataset_sources_for_train(). This is mandatory.
    def clean_data_for_train(self, datasets: dict) -> pandas.DataFrame:
        pass


    # Implement this method if you need to modify data integrated by
    # clean_data_for_train().
    # Must return a DataFrame that has at least x_estimator_features columns.
    # def feature_engineering_for_train(self, data: pandas.DataFrame) -> pandas.DataFrame:
    #     return data


    # Implement this method if you need a last chance to process data returned
    # by feature_engineering_for_train() and right before training.
    # def last_pre_process_for_train(self, data: pandas.DataFrame) -> pandas.DataFrame:
    #     return data



    # Same as sibling methods above, but for batch predict data. The clean...()
    # is mandatory if you have batch predict in your pipeline
    # def clean_data_for_batch_predict(self, datasets: dict) -> pandas.DataFrame:
    #     return pandas.DataFrame()
    # def feature_engineering_for_batch_predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
    #     return data
    # def last_pre_process_for_batch_predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
    #     return data

    
    
    
    # This is the standard method that should be called if you need to convert
    # x_features into x_estimator_features. Probably your
    # feature_engineering_for_train() and feature_engineering_for_batch_predict()
    # methods will want to call this method.
    # Must return a DataFrame that has at least x_estimator_features columns.
    def feature_engineering_for_predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        return data