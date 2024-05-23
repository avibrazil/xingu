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
import pydantic

import xingu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(''), '..')))

import estimators.xgboost_classifier


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

    api_router = {
        '/will_survive': (api_predict_survive)
    }

    encode_cols = ['Sex','Embarked']

    # 1=child, 2=adult, 3=elder, 4=unknown
    age_ranges     = [0,    15,      50,      100,      numpy.inf]
    age_categories = [   1,       2,       3,       4            ]


    estimator_class = estimators.xgboost_classifier.XinguXGBoostClassifier

    ## XGBoost with Optuna...
    estimator_class_params = dict(
        # Número de splits e número de algoritmos que serão treinados
        bagging_size            = 3,

        # A cada quantos segundos o otimizador e pareto-front incompletos
        # são salvos
        report_interval         = 30,

        # Número de iterações de otimização. Cada iteração
        # treina {bagging_size} XGBoosts
        optimization_trials     = 3000,

        # Tempo máximo de otimização em segundos
        optimization_timeout    = 5*3600,
    )

    estimator_params = dict(
        n_jobs                  = -1,
        objective               = 'binary:logistic',
        eval_metric             = 'logloss',
        use_label_encoder       = False,
        missing                 = numpy.nan,
    )

    # Search space for Optuna
    estimator_hyperparams_search_space = dict(
        n_estimators            = ('int',        dict(low=10,    high=500)),
        alpha                   = ('float',      dict(low=1e-3,  high=10)),
        gamma                   = ('float',      dict(low=1e-3,  high=10)),
        colsample_bytree        = ('categorical',dict(choices=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
        subsample               = ('categorical',dict(choices=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
        learning_rate           = ('float',      dict(low=0.01,  high=0.05)),
        max_depth               = ('int',        dict(low=3,     high=7)),
        min_child_weight        = ('int',        dict(low=1,     high=10)),
        **{
            'lambda'            : ('float',     dict(low=1e-3,  high=10)),
        }
    )

    # Parameters computed from an optimization optuna's genetic algorithms
    # [Val AUC, Train AUC-Val AUC] = [0.863929889298893, 0.006449178128144939]
    estimator_hyperparams = {
        'n_estimators': 18,
        'alpha': 0.6976961980825642,
        'gamma': 0.5000088656846183,
        'colsample_bytree': 0.8,
        'subsample': 1.0,
        'learning_rate': 0.03708621167071816,
        'max_depth': 3,
        'min_child_weight': 2,
        'lambda': 4.920621243793648
    }

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
    ##  API methods
    ##
    ##
    ###########################################################################

    def api_predict_survive(
            self,
            request_params: pydantic.create_model ,
            model
        ):


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
