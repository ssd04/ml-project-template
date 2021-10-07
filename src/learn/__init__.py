from .classification.randomforest import RandomForest
from .classification.xgboost import XGBoost
from .regression.logistic import LogisticR
from .regression.xgboost import XGBoostR


class GetModel:
    @classmethod
    def get_model(cls, alg, conf=None):
        if alg == "random_forest":
            model = RandomForest(conf=conf)
        elif alg == "logistic_regression":
            model = LogisticR(conf=conf)
        elif alg == "dummy_classifier":
            model = Dummy(conf=conf)
        elif alg == "xgboost":
            model = XGBoost(conf=conf)
        elif alg == "xgboost_regressor":
            model = XGBoostR(conf=conf)
        else:
            raise ValueError(alg)
        return model
