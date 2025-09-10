import os
import sys
import datetime
import re
import logging
import pickle
import pathlib
import concurrent.futures
import numpy
import matplotlib
import pandas
import sklearn.preprocessing
import sklearn.model_selection
import xingu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(''), '..')))

import xingu.estimators.xgboost_optuna


class DPTitanicSurvivor(xingu.DataProvider):
    id = 'titanic'

    # Name of target column in the train dataset
    y = 'Survived'

    # Plain .predict() yields class 1 (survived), not class 0 (dead).
    proba_class_index = 1

    x_features = """
        Pclass
        Sex
        Age
        SibSp
        Parch
        Fare
        Embarked
    """.split()

    # Some of these columns will be engineered in generic_feature_engineering()
    x_estimator_features = """
        Pclass
        Age_encoded
        SibSp
        Parch
        Fare
        Sex_encoded
        Embarked_encoded
        FamilySize
        IsAlone
    """.split()

    encode_cols = ['Sex','Embarked']

    # 1=child, 2=adult, 3=elder, 4=unknown
    age_ranges     = [0,    15,      50,      100,      numpy.inf]
    age_categories = [   1,       2,       3,       4            ]


    estimator_class = xingu.estimators.xgboost_optuna.XinguXGBoostClassifier

    # XGBoost with Optuna: parameters for
    # xingu.estimators.xgboost_optuna.XinguXGBoostClassifier class....
    # This is an ensamble of 3 XGBoosts; the optimizer will try 2500 different
    # combinations of hyperparameters or run for maximum of 5 hours
    estimator_class_params = dict(
        # Number of cross validation splits and number of XGBoosts that will
        # be trained
        bagging_size                  = 3,

        # Number of optimization interations. Each interation
        # trains {bagging_size} XGBoosts
        optimization_trials           = 2500,

        # Maximum optimization time in seconds
        optimization_timeout          = 5*3600,

        # Time interval in seconds on which the optimizer incomplete
        # pareto-front graph will be saved
        optimization_report_interval  = 30,
    )

    # XGBoost initialization parameters
    # Notice we are optimizing hyperparameters (and training) using NVidia
    # Cuda (GPU), which performs 10 times faster than CPU.
    # If you optimize hyperparameters with GPU, you have to train your model
    # with GPU too.
    estimator_params = dict(
        n_jobs                  = -1,
        objective               = 'binary:logistic',
        eval_metric             = 'logloss',
        missing                 = numpy.nan,
        verbose                 = False,
        device                  = 'cuda',
    )

    # Search space for Optuna. Optimization objective is to find best combined
    # values inside these ranges
    estimator_hyperparams_search_space = dict(
        n_estimators            = ('int',         dict(low=10,    high=500)),
        alpha                   = ('float',       dict(low=1e-3,  high=10)),
        gamma                   = ('float',       dict(low=1e-3,  high=10)),
        colsample_bytree        = ('categorical', dict(choices=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
        subsample               = ('categorical', dict(choices=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
        learning_rate           = ('float',       dict(low=0.01,  high=0.05)),
        max_depth               = ('int',         dict(low=3,     high=7)),
        min_child_weight        = ('int',         dict(low=1,     high=10)),
        **{
            'lambda'            : ('float',       dict(low=1e-3,  high=10)),
        }
    )

    # Parameters computed from an optimization optuna's genetic algorithms
    # [Validation_AUC, Train_AUC-Validation-AUC] = [0.863929889298893, 0.006449178128144939]
    estimator_hyperparams = dict(
        alpha            = 9.646828530085509,
        colsample_bytree = 5,
        gamma            = 9.652085454387114,
        lambda           = 5.244338280711388,
        learning_rate    = 0.025889435953364674,
        max_depth        = 4,
        min_child_weight = 4,
        n_estimators     = 11,
        subsample        = 0,
    )

    # Data need to be downloaded manually from https://www.kaggle.com/competitions/titanic/data
    train_dataset_sources = dict(
        train = dict(
            url = 'data/train.csv',
        ),
    )

    batch_predict_dataset_sources = dict(
        train = dict(
            url = 'data/test.csv',
        ),
    )



    def clean_data_for_train(self, datasets: dict) -> pandas.DataFrame:
        return self.clean_data_for_batch_predict(datasets)



    def clean_data_for_batch_predict(self, datasets: dict) -> pandas.DataFrame:
        """
        Mission is to integrate all dataframes in the datasets dict and do any
        cleanup needed that is not feature engineering.
        """
        return (
            pandas.concat(
                [
                    datasets[d]
                    for d in datasets.keys()
                ]
            )
            .set_index('PassengerId')
        )



    def feature_engineering_for_train(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
        Compute some column encoders and save them in the object for later use.
        Then do the feature engineering tasks.
        """
        self.encoders={
            col: sklearn.preprocessing.OrdinalEncoder().fit(df[[col]].dropna())
            for col in self.encode_cols
        }

        def ddebug(table,message):
            self.log(message, level=logging.DEBUG)
            return table

        return (
            df

            # Remove rows with NaNs on some columns
            .dropna(subset=self.encode_cols)

            # Create a column named split with labels "train" and "test"
            .pipe(
                lambda table: table.join(
                    table
                    .sample(frac=0.2,random_state=42)
                    .assign(split='test')
                    .split,
                    how='left'
                )
            )
            .assign(
                # Optimize the split column
                split=lambda table: table.split.fillna('train').astype('category'),

                # Our estimator requires a "stratify" column to use with its
                # StratifiedKFold validation method
                stratify = lambda table: table[self.y],
            )

            # Pass table through our generic feature engineering
            .pipe(self.generic_feature_engineering)

            .pipe(
                lambda table: ddebug(
                    table,
                    "Engineered table:\n" + table[self.x_estimator_features].head(10).to_markdown()
                )
            )
        )



    def feature_engineering_for_batch_predict(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
        Use pre-computed encoders to transform categorical columns.
        """
        return self.generic_feature_engineering(df)



    def generic_feature_engineering(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
        Input DataFrame is expected to have at least columns on x_features.
        Makes transformations and returns a new DataFrame with at least
        x_estimator_features columns.

        This method should be called before .predict() or .predict_proba().
        The xingu.Model class calls it via
        feature_engineering_for_batch_predict() or feature_engineering_for_train()
        but on a regular API you must call it explicitly like this:

        output=my_titanic_model.dp.generic_feature_engineering(input)
        y_pred=my_titanic_model.predict(output)
        """

        return (
            df
            .assign(
                Sex_encoded      = lambda table: (
                    self.encoders['Sex']
                    .transform(table[['Sex']])
                    .astype(int)
                ),

                Embarked_encoded = lambda table: (
                    self.encoders['Embarked']
                    .transform(table[['Embarked']])
                    .astype(int)
                ),

                Age_encoded      = lambda table: pandas.cut(
                    x=table.Age.fillna(numpy.inf),
                    bins=self.age_ranges,
                    labels=self.age_categories
                ).astype(int),

                # Create a feature for family size
                FamilySize = lambda table: table.SibSp + table.Parch + 1,

                # Create a binary feature for whether the passenger is traveling alone
                IsAlone = lambda table: (table.FamilySize == 1).astype(int),
            )
        )



    def data_split_for_train(self, data: pandas.DataFrame) -> dict:
        return {
            s: data.query(f"split=='{s}'")
            for s in data.split.unique()
        }


    def post_process_after_hyperparam_optimize(self, model):
        """
        Chamado múltiplas vezes durante a otimização de hyperparâmetros e
        também logo após o fim da otimização. Este método faz 2 coisas:

        - Salva gráfico Pareto-front em formato HTML+Plotly (https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_pareto_front.html)
        - Salva PKL do objeto xingu.xgboost_optuna.XinguXGBoostClassifier.optimizer

        Caso o treino já tenha finalizado, os nomes dos arquivos conterão data
        e hora do fim do treino e serão apagados os arquivos intermediários
        anteriores.
        """
        plot_template_incomplete      = "{dp} • {full_train_id} • global • Pareto-front.html"
        plot_template                 = "{dp} • {time} • {full_train_id} • global • Pareto-front.html"
        optimizer_template_incomplete = '{dp} • {full_train_id} • optimizer.pkl'
        optimizer_template            = '{dp} • {time} • {full_train_id} • optimizer.pkl'

        if hasattr(model.estimator,'optimizer_pareto_front'):
            has_time = type(model.trained) is datetime.datetime
            if has_time:
                # Train has finished; set final template and remove transient file
                tpl = plot_template

                (
                    pathlib.Path(model.get_config('PLOTS_PATH', default='.')) /
                    plot_template_incomplete.format(
                        dp=self.id,
                        full_train_id=model.get_full_train_id(),
                    )
                ).unlink(missing_ok=True)
            else:
                # Unfinished train; use a template without train time
                tpl = plot_template_incomplete

            # Write transient or final file
            model.estimator.optimizer_pareto_front.write_html(
                pathlib.Path(model.get_config('PLOTS_PATH', default='.')) /
                tpl.format(
                    dp=self.id,
                    full_train_id=model.get_full_train_id(),
                    time=(
                        type(model).time_fs_str(model.trained)
                        if has_time
                        else datetime.datetime.now()
                    ),
                )
            )

        if hasattr(model.estimator,'optimizer'):
            has_time = type(model.trained) is datetime.datetime
            if has_time:
                # Train has finished; set final template and remove transient file
                tpl=optimizer_template

                (
                    pathlib.Path(model.get_config('TRAINED_MODELS_PATH', default='.')) /
                    optimizer_template_incomplete.format(
                        dp=self.id,
                        full_train_id=model.get_full_train_id(),
                    )
                ).unlink(missing_ok=True)
            else:
                # Unfinished train; use a template without train time
                tpl=optimizer_template_incomplete

            pkl=open(
                pathlib.Path(model.get_config('TRAINED_MODELS_PATH', default='.')) /
                tpl.format(
                    dp=self.id,
                    full_train_id=model.get_full_train_id(),
                    time=(
                        type(model).time_fs_str(model.trained)
                        if has_time
                        else datetime.datetime.now()
                    ),
                ),
                'wb'
            )
            try:
                pickle.dump(model.estimator.optimizer, pkl)
            except RuntimeError:
                # A "RuntimeError: dictionary changed size during iteration"
                # might happen here because the model.estimator.optimizer
                # object is still alive and hot in the background. Simply
                # ignore it, tell user and try again on next iteration.
                self.log(
                    level=logging.WARNING,
                    message="Optimizer has changed in the background. Will try again on next cycle"
                )
            pkl.close()



    def post_process_after_train(self, model):
        """
        Chamado logo após o fim do treino, este método
        faz 2 coisas:

        - Salva gráfico Pareto-front em formato HTML+Plotly (https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_pareto_front.html)
        - Salva PKL do objeto xingu.xgboost_optuna.XinguXGBoostClassifier.optimizer
        - TODO: Calcula segmentos de score
        """
        self.post_process_after_hyperparam_optimize(model)



    ###########################################################################
    ##
    ##  Operational methods
    ##
    ##
    ###########################################################################

    def __getstate__(self):
        return dict(
            **super().__getstate__(),

            encoders             = self.encoders,
            encode_cols          = self.encode_cols,
        )
