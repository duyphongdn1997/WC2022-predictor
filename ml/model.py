import time
from typing import Text

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, plot_confusion_matrix, roc_curve, \
    classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from configs.constants import SUPPORT_MODEL, DEFAULT_MODEL
from ml.data_prepare import data_preparing, create_dataset


def plot_roc_cur(fper, tper):
    """
    PLot the ROC
    :param fper:
    :param tper:
    """
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


class MLModel:
    """
    WC predictor model
    """

    def __init__(self, model_type: Text):

        assert model_type in SUPPORT_MODEL, \
            "Not support the kind of model. Please choose one of {}".format(SUPPORT_MODEL)
        self.model_type = model_type
        if self.model_type == "LogisticRegression":
            self.model = self.get_logistic_regression_model()
        elif self.model_type == "DecisionTreeClassifier":
            self.model = self.get_decision_tree_model()
        elif self.model_type == "MLPClassifier":
            self.model = self.get_neural_network_model()
        elif self.model_type == "RandomForestClassifier":
            self.model = self.get_random_forest_model()
        elif self.model_type == "GradientBoostingClassifier":
            self.model = self.get_gradient_boosting_model()
        elif self.model_type == "LGBMClassifier":
            self.model = self.get_light_gbm_model()
        elif self.model_type == "XGBClassifier":
            self.model = self.get_xgboost_model()

    def predict_proba(self, x):
        """
        Call predict_proba on the estimator with the best found parameters.
        :return:
        """
        return self.model.predict_proba(x)

    @staticmethod
    def __run_model(model, x_train, y_train, x_test, y_test, verbose=True):
        t0 = time.time()
        if verbose is False:
            model.fit(x_train.values, np.ravel(y_train), verbose=0)
        else:
            model.fit(x_train.values, np.ravel(y_train))
        model = model.best_estimator_
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test.values, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(x_test.values)[:, 1])
        coh_kap = cohen_kappa_score(y_test, y_pred)
        time_taken = time.time() - t0
        print("Accuracy : {}".format(accuracy))
        print("ROC Area under Curve : {}".format(roc_auc))
        print("Cohen's Kappa : {}".format(coh_kap))
        print("Time taken : {}".format(time_taken))
        print(classification_report(y_test, y_pred, digits=5))

        return model, accuracy, roc_auc, coh_kap, time_taken

    @staticmethod
    def get_logistic_regression_model(**params_lr):
        """
        Return a logistic regression model
        :return:
        """
        if not all(params_lr.values()):
            params_lr = {
                "C": np.logspace(-3, 3, 7),
                "penalty": ["l1", "l2"],
                'solver': 'liblinear'
            }

        model_lr = LogisticRegression()
        model_lr = GridSearchCV(model_lr, params_lr, cv=3, verbose=False, scoring='roc_auc', refit=True)
        return model_lr

    @staticmethod
    def get_decision_tree_model(**params):
        """
        Return a decision tree model
        :return:
        """
        if not all(params.values()):
            params = {'max_features': ['auto', 'sqrt', 'log2'],
                      'ccp_alpha': [0.1, .01, .001],
                      'max_depth': [5, 6, 7, 8, 9],
                      'criterion': ['gini', 'entropy']
                      }

        model = DecisionTreeClassifier()
        model = GridSearchCV(estimator=model, param_grid=params, cv=3, verbose=False, scoring='roc_auc', refit=True)
        return model

    @staticmethod
    def get_neural_network_model(**params_nn):
        """
        Return a neutral network model
        :return:
        """
        if not all(params_nn.values()):
            params_nn = {'solver': ['lbfgs'],
                         'max_iter': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
                         'alpha': 10.0 ** -np.arange(1, 10),
                         'hidden_layer_sizes': np.arange(10, 15),
                         'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}

        model_nn = MLPClassifier()
        model_nn = GridSearchCV(model_nn, params_nn, n_jobs=-1, scoring='roc_auc', refit=True, verbose=False)
        return model_nn

    @staticmethod
    def get_random_forest_model(**params_rf):
        """
        Return a random forest model
        :return:
        """
        if not all(params_rf.values()):
            params_rf = {"max_depth": [20],
                         "min_samples_split": [10],
                         "max_leaf_nodes": [175],
                         "min_samples_leaf": [5],
                         "n_estimators": [250],
                         "max_features": ["sqrt"],
                         }

        model_rf = RandomForestClassifier()
        model_rf = GridSearchCV(model_rf, params_rf, cv=3, n_jobs=-1, verbose=False, scoring='roc_auc', refit=True)

        return model_rf

    @staticmethod
    def get_light_gbm_model(**params_lgb):
        """
        Return a LightGBM model
        :return:
        """
        if not all(params_lgb.values()):
            params_lgb = {
                'learning_rate': [0.005, 0.01],
                'n_estimators': [8, 16, 24],
                'num_leaves': [6, 8, 12, 16],  # large num_leaves helps improve accuracy but might lead to over-fitting
                'boosting_type': ['gbdt', 'dart'],  # for better accuracy -> try dart
                'objective': ['binary'],
                'max_bin': [255, 510],  # large max_bin helps improve accuracy but might slow down training progress
                'random_state': [500],
                'colsample_bytree': [0.64, 0.65, 0.66],
                'subsample': [0.7, 0.75],
                'reg_alpha': [1, 1.2],
                'reg_lambda': [1, 1.2, 1.4],
            }

        model = lgb.LGBMClassifier()
        model = GridSearchCV(model, params_lgb, verbose=False, cv=3, n_jobs=-1, scoring='roc_auc', refit=True)

        return model

    @staticmethod
    def get_xgboost_model(**params_xgb):
        """
        Return a xgboost model
        :return:
        """
        if not all(params_xgb.values()):
            params_xgb = {
                'nthread': [4],  # when use hyper thread, xgboost may become slower
                'objective': ['binary:logistic'],
                'learning_rate': [0.05],  # so called `eta` value
                'max_depth': [6],
                'min_child_weight': [11],
                'silent': [1],
                'subsample': [0.8],
                'colsample_bytree': [0.7],
                'n_estimators': [100],  # number of trees, change it to 1000 for better results
                'missing': [-999],
                'seed': [1337]
            }
        model = GridSearchCV(xgb.XGBClassifier(), params_xgb, n_jobs=-1,
                             cv=3,
                             scoring='roc_auc',
                             refit=True)

        return model

    def fit_and_eval_model(self, x_train, x_test, y_train, y_test):
        """
        Run the model with dataset
        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :return:
        """
        model_lr, accuracy_lr, roc_auc_lr, coh_kap_lr, tt_lr = \
            self.__run_model(self.model, x_train, y_train, x_test, y_test)
        return model_lr, accuracy_lr, roc_auc_lr, coh_kap_lr, tt_lr

    @staticmethod
    def get_gradient_boosting_model(**params):
        """
        Return gradient boosting model
        :param params:
        :return:
        """
        if not all(params.values()):
            params = {"learning_rate": [0.01, 0.02, 0.03],
                      "min_samples_split": [5, 10],
                      "min_samples_leaf": [3, 5],
                      "max_depth": [3, 5, 10],
                      "max_features": ["sqrt"],
                      "n_estimators": [100, 200]
                      }
        model = GradientBoostingClassifier(random_state=100)
        return GridSearchCV(model, params, cv=3, n_jobs=-1)


base_df, data_df = data_preparing()
x_train, x_test, y_train, y_test = create_dataset(data_df)
ml_model = MLModel(DEFAULT_MODEL)
ml_model.fit_and_eval_model(x_train, x_test, y_train, y_test)
