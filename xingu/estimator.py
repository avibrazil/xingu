import logging
import numpy
import os
import pandas as pd
from typing import List



class Estimator(object):
    """
    Base class to encapsulate some very specific features of Loft AVM estimators:

    - Concrete class might have multiple (20) encapsulated predictors for bagging/ensamble.
    - pred_dist() must return μ and σ of a Gaussian distribution. If concrete class has
      multiple estimators, an additional score per-unit is returned.
    - pred_quantiles() returns all that pred_dist() returns plus values from p_05 to p_95.
      With p_50 the same as μ from pred_dist().
    """


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


    hyperparam = dict()



    def __init__(self):
        self.setup_logger()



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



    def hyperparam_optimize(self, train: pd.DataFrame, val: pd.DataFrame, features: List[str], target: str):
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



    def fit(self, train, val, features, target):
        """
        Train one or multiple models. Leave everything ready for a predict()
        """
        pass



    def pred_dist(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Return a dataframe with estimated loc (μ), scale (σ) and score (in case of
        multi-models) indexed by unit_id.

        Columns returned:
        - unit_id (index)
        - loc (estimated price per m² in monetary units)
        - scale (m² price standard deviation in monetary units)
        - score (1-mean(scale)÷mean(loc)) (valid only for bagging/ensamble estimators)
        """
        pass



    def pred_quantiles(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a dataframe with estimated p_05..p_95 and score (in case of
        multi-models) indexed by unit_id.

        Columns returned:
        - unit_id (index)
        - p_05..p_95 (estimated price per m² in monetary units, for each quantile)
        - scale (m² p_50 price standard deviation in monetary units)
        - score (1-mean(scale)÷mean(p_50)) (valid only for bagging/ensamble estimators)
        """
        pass


    def __repr__(self):
        return '{klass}()'.format(klass=type(self).__name__)



    def __getstate__(self):
        # Do not serialize the logger
        return dict(
            QUANTILES       = self.QUANTILES,
            LOWER_BOUND     = self.LOWER_BOUND,
            UPPER_BOUND     = self.UPPER_BOUND,
            POINT_ESTIMATE  = self.POINT_ESTIMATE,
            hyperparam      = self.hyperparam,
        )










