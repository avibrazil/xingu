# Xingu for automated ML model training

Xingu is a framework of a few classes that helps on full industrialization of
your machine learning training and deployment pipelines. Just write your
DataProvider class, mostly in a declarative way, that completely controls
your training and deployment pipeline.

Notebooks are useful in EDA time, but when the modeling is ready to become
a product, use Xingu proposed classes to organize interactions with DB
(queries), data cleanup, feature engineering, hyper-parameters optimization,
training algorithm, general and custom metrics computation, estimation
post-processing.

- Don’t save a pickle at the end of your EDA, let Xingu organize a versioned
  inventory of saved models (PKLs) linked and associated to commit hashes and
  branches of your code.

- Don’t save metrics manually and in an informal way. Metrics are first class
  citizens, so use Xingu to write methods that compute metrics and let it
  store metrics in an organized database that can be queried and compared.

- Don’t make ad-hoc plots to understand your data. Plots are important assets
  to measure the quality of your model, so use Xingu to write methods that
  formaly generate versioned plots.

- Do not worry or even write code that loads pre-req models, use Xingu pre-req
  architecture to load pre-req models for you and package them together.

- Don’t save ad-hoc hypermaters after optimizations. Let Xingu store and manage
  those for you in a way that can be reused in future trains.
  
- Don’t change your code if you want different functionality. Use Xingu
  environment variables or command line parameters to strategize your trains.

- Don’t manually copy PKLs to production environments on S3 or other object
  storage. Use Xingu’s deployment tools to automate the deployment step.
  
- Don’t write database integration code. Just provide your queries and Xingu
  will give you the data. Xingu will also maintain a local cache of your data
  so you won’t hammer your database across multiple retrains. Do the same with
  static data files with parquet, CSV, on local filesystem or object storage.
  
- Xingu can run anyware, from your laptop, with a plain SQLite database, to
  large scale cloud-powered training pipelines with GitOps, Jenkins, Docker
  etc. Xingu’s database is used only to cellect training information, it isn´t
  required later when model is used to predict.

## Install
```shell
pip install https://github.com/avibrazil/xingu
```

or

```shell
pip install xingu
```

## Use to Train a Model
Check your project has the necessary files and folders:
```shell
$ find
dataproviders/
dataproviders/my_dataprovider.py
estimators/
estimators/myrandomestimator.py
models/
data/
plots/
```
Train with DataProviders `id_of_my_dataprovider1` and `id_of_my_dataprovider2`, both defined in `dataproviders/my_dataprovider.py`:
```shell
$ xingu \
    --dps id_of_my_dataprovider1,id_of_my_dataprovider2 \
    --databases athena "awsathena+rest://athena.us..." \
    --query-cache-path data \
    --trained-models-path models \
    --debug
```

## Use the API
See the [proof of concept notebook](https://github.com/avibrazil/xingu/blob/main/notebooks/POC%20Use%20Xingu.ipynb) with vairous usage scenarios:

- POC 1. Train some Models
- POC 2. Use Pre-Trained Models for Batch Predict
- POC 3. Assess Metrics and create Comparative Reports
- POC 4. Check and report how Metrics evolved
- POC 5. Play with Xingu barebones
- POC 6. Play with the `ConfigManager`
- POC 7. Xingu Estimators in the Command Line
- POC 8. Deploy Xingu Data and Estimators between environments (laptop, staging, production etc)

## Procedures defined by Xingu

Xingu classes do all the heavy lifting while you focus on your machine learning
code only.

- Class `Coach` is responsible of coordinating the training process of one or
multiple models. You control parallelism via command line or environment
variables.

- Class `Model` implements a standard pipelines for train, train with hyperparam
optimization, load and save pickles, database access etc. These pipelines are
is fully controlled by your DataProvider or the environment.

- Class `DataProvider` is a base class that is constantly queried by the `Model`
to determine how the `Model` should operate. Your should create a class derived
from `DataProvider` and reimplement whatever you want to change. This will
completely change behaviour of `Model` operation in a way that you´ll get a
completelly different model.

    - It is your `DataProvider` that defines the source of training data as SQL
    queries or URLs of parquets, CSVs, JSONs
    - It is your `DataProvider` that defines how multi-source data should be
    integrated
    - It is your `DataProvider` that defines how data should be split into train
    and test sets
    - Your `DataProvider` defines which `Estimator` class to use
    - Your `DataProvider` defines how the `Estimator` should be initialized and
    optimized
    - Your `DataProvider` defines which metrics should be computed, how to
    compute them and against which dataset
    - Your `DataProvider` defines which plots should be created and against
    which dataset
    - See below when and how each method of your `DataProvider` will be called
    by `xingu.Model`
    
- Class `Estimator` is another base class (that you can reimplement) to contain
estimator-specific affairs. There will be an `Estimator`-derived class for an
XGBoostRegressor, other for a CatBoostClassifier, other for a
SciKit-Learn-specific algorithm, including hyperparam optimization logic and
libraries. A concrete `Estimator` class can and should be reused across multiple
different models.

The hierarchical diagrams below expose complete Xingu pipelines with all their
steps. Steps marked with 💫 are were you put your code. All the rest is Xingu
boilerplate code ready to use.

### `Coach.team_train()`:

Train various Models, all possible in parallel.

1. `Coach.team_train_parallel()` (background, parallelism controled by `PARALLEL_TRAIN_MAX_WORKERS`):
    1. `Coach.team_load()` (for pre-req models not trained in this session)
    2. Per DataProvider requested to be trained:
        1. `Coach.team_train_member()` (background):
            1. `Model.fit()` calls:
                1. 💫`DataProvider.get_dataset_sources_for_train()` return dict of queries and/or URLs
                2. `Model.data_sources_to_data(sources)`
                3. 💫`DataProvider.clean_data_for_train(dict of DataFrames)`
                4. 💫`DataProvider.feature_engineering_for_train(DataFrame)`
                5. 💫`DataProvider.last_pre_process_for_train(DataFrame)`
                6. 💫`DataProvider.data_split_for_train(DataFrame)` return tuple of dataframes
                7. `Model.hyperparam_optimize()` (decide origin of hyperparam)
                    1. 💫`DataProvider.get_estimator_features_list()`
                    2. 💫`DataProvider.get_target()`
                    3. 💫`DataProvider.get_estimator_optimization_search_space()`
                    4. 💫`DataProvider.get_estimator_hyperparameters()`
                    5. 💫`Estimator.hyperparam_optimize()` (SKOpt, GridSearch et all)
                    6. 💫`Estimator.hyperparam_exchange()`
                9. 💫`Estimator.fit()`
                10. 💫`DataProvider.post_process_after_train()`
    2. `Coach.post_train_parallel()` (background, only if `POST_PROCESS=true`):
        1. Per trained Model (parallelism controled by `PARALLEL_POST_PROCESS_MAX_WORKERS`):
            1. `Model.save()` (PKL save in background)
            2. `Model.trainsets_save()` (save the train datasets, background)
            3. `Model.trainsets_predict()`:
                1. `Model.predict_proba()` or `Model.predict()` (see [below](#predict))
                2. `Model.compute_and_save_metrics(channel=trainsets)` (see [below](#metrics))
            4. `Coachl.single_batch_predict()` (see [below](#batch))



<a id='batch'></a>
### `Coach.team_batch_predict()`:

Load from storage and use various pre-trained Models to estimate data from a pre-defined SQL query.
The batch predict SQL query is defined into the DataProvider and this process will query the database
to get it.

1. `Coach.team_load()` (for all requested DPs and their pre-reqs)
2. Per loaded model:
    1. `Coach.single_batch_predict()` (background)
        1. `Model.batch_predict()`
            1. 💫`DataProvider.get_dataset_sources_for_batch_predict()`
            2. `Model.data_sources_to_data()`
            3. 💫`DataProvider.clean_data_for_batch_predict()`
            4. 💫`DataProvider.feature_engineering_for_batch_predict()`
            5. 💫`DataProvider.last_pre_process_for_batch_predict()`
            6. `Model.predict_proba()` or `Model.predict()` (see [below](#predict))
        2. `Model.compute_and_save_metrics(channel=batch_predict` (see [below](#metrics))
        3. `Model.save_batch_predict_estimations()`


<a id='predict'></a>
### `Model.predict()` and `Model.predict_proba()`:

1. `Model.generic_predict()`
    1. 💫`DataProvider.pre_process_for_predict()` or `DataProvider.pre_process_for_predict_proba()`
    2. 💫`DataProvider.get_estimator_features_list()`
    3. 💫`Estimator.predict()` or `Estimator.predict_proba()`
    4. 💫`DataProvider.post_process_after_predict()` or `DataProvider.post_process_after_predict_proba()`


<a id='metrics'></a>
### `Model.compute_and_save_metrics()`:

Sub-system to compute various metrics, graphics and transformations over
a facet of the data.

This is executed right after a Model was trained and also during a batch predict.

Predicted data is computed before `Model.compute_and_save_metrics()` is called.
By `Model.trainsets_predict()` and `Model.batch_predict()`.

1. `Model.save_model_metrics()` calls:
    1. `Model.compute_model_metrics()` calls:
        1. `Model.compute_trainsets_model_metrics()` calls:
            1. All `Model.compute_trainsets_model_metrics_{NAME}()`
            2. All 💫`DataProvider.compute_trainsets_model_metrics_{NAME}()`
        2. `Model.compute_batch_model_metrics()` calls:
            1. All `Model.compute_batch_model_metrics_{NAME}()`
            2. All 💫`DataProvider.compute_batch_model_metrics_{NAME}()`
        3. `Model.compute_global_model_metrics()` calls:
            1. All `Model.compute_global_model_metrics_{NAME}()`
            2. All 💫`DataProvider.compute_global_model_metrics_{NAME}()`
    2. `Model.render_model_plots()` calls:
        1. `Model.render_trainsets_model_plots()` calls:
            1. All `Model.render_trainsets_model_plots_{NAME}()`
            3. All 💫`DataProvider.render_trainsets_model_plots_{NAME}()`
        2. `Model.render_batch_model_plots()` calls:
            1. All `Model.render_batch_model_plots_{NAME}()`
            3. All 💫`DataProvider.render_batch_model_plots_{NAME}()`
        3. `Model.render_global_model_plots()` calls:
            1. All `Model.render_global_model_plots_{NAME}()`
            3. All 💫`DataProvider.render_global_model_plots_{NAME}()`
2. `Model.save_estimation_metrics()` calls:
    1. `Model.compute_estimation_metrics()` calls:
        1. All `Model.compute_estimation_metrics_{NAME}()`
        2. All 💫`DataProvider.compute_estimation_metrics_{NAME}()`
