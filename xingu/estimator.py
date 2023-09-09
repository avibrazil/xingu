import os
import logging
import pandas



class Estimator(object):
    """
    Base class to encapsulate an estimator as defined by SciKit-Learn

    - Concrete class might have multiple (20) encapsulated predictors for bagging/ensamble.
    """

    
    hyperparam = dict()



    def __init__(self,hyperparams=None):
        self.setup_logger()
        self.hyperparam=hyperparams



    def hyperparam_optimize(self, datasets: dict, features: list, target: str, search_space: dict) -> dict:
        # Please implement in a concrete class
        pass



    def hyperparam_exchange(self, hyperparam: dict=None) -> dict:
        """
        If None is passed, simply return estimator's current hyperparam.
        Set hyperparam otherwise.
        """

        if hyperparam is not None:
            self.hyperparam=hyperparam

        return self.hyperparam



    def fit(self, datasets: dict, features: list, target: str):
        """
        Train one or multiple models. Leave everything ready for a predict()
        """
        pass



    def predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        """
        Return a dataframe with estimated Ŷ and indexed as input DataFrame.
        """
        pass



    def predict_proba(self, data: pandas.DataFrame, class_index: int=None) -> pandas.DataFrame:
        """
        Estimate probability of all classes.
        If class_index is not None, only the class with that index will be returned.

        Return a dataframe with estimated Ŷ and indexed as input DataFrame.
        """
        pass



    def is_classifier(self):
        pass
    
    
    
    def __repr__(self):
        return '{klass}()'.format(klass=type(self).__name__)



    def __getstate__(self):
        # Do not serialize the logger
        return dict(
            hyperparam      = self.hyperparam,
        )



    def setup_logger(self):
        if not hasattr(self,'logger'):
            # Setup logging
            self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)



    def log(self, message='', level=logging.INFO):
        self.setup_logger()
        self.logger.log(
            level,
            message
        )









