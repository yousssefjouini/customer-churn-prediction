import pandas as pd
import xgboost as xgb
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    f1_score,
)
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import confusion_matrix
import seaborn as sns

import logging

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


class ChurnClassifier:
    def __init__(self, data):
        self.data = data
        self.threshold = 0.5

    # def _prepare_data(self):
    #     # Split the data into features and target
    #     self.X = self.data.drop(['churn', 'client_id','recency','r_score','cluster'], axis=1)
    #     self.y = self.data['churn']

    #     # Split the data into training and testing sets
    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=5)

    def _prepare_data(self):
        # Split the data into training and testing sets
        self.train_data, self.test_data = train_test_split(
            self.data, test_size=0.2, random_state=42
        )
        # Split the data into features and target

        self.X_train, self.y_train = (
            self.train_data.drop(
                ["churn", "client_id", "recency", "r_score", "cluster"], axis=1
            ),
            self.train_data["churn"],
        )
        self.X_test, self.y_test = (
            self.test_data.drop(
                ["churn", "client_id", "recency", "r_score", "cluster"], axis=1
            ),
            self.test_data["churn"],
        )

    def smote_data(self):
        self.X_train, self.y_train = SMOTE().fit_resample(self.X_train, self.y_train)

    def train(self, model, smote=False):
        self._prepare_data()
        # Train an XGBoost classifier with default hyperparameters
        if smote:
            self.smote_data()
        self.model = model
        self.model.fit(self.X_train, self.y_train)

    def select_threshold(self):
        fpr, tpr, thresholds = roc_curve(
            self.y_test, self.model.predict_proba(self.X_test)[:, 1]
        )
        optimal_idx = np.argmax(tpr - fpr)
        self.threshold = thresholds[optimal_idx]
        print("Optimal threshold:", self.threshold)
        return self.threshold

    def predict(self):
        self.y_pred = (
            self.model.predict_proba(self.X_test)[:, 1] >= self.threshold
        ).astype(int)

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt="g")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def evaluate(self):
        self.predict()
        f1 = f1_score(self.y_test, self.y_pred)
        print(f"F1 score: {f1:.4f}")
        print(classification_report(self.y_test, self.y_pred))
        self.plot_confusion_matrix()
        plt.show()

    def auc(self):
        print(
            "AUC score:",
            roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1]),
        )
        fpr, tpr, thresholds = roc_curve(
            self.y_test, self.model.predict_proba(self.X_test)[:, 1]
        )
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.show()

    def plot_feature_importance(self):
        # Plot feature importance
        xgb.plot_importance(self.model)

    def shapley_values(self):
        explainer = shap.Explainer(self.model)
        shap_values = explainer(self.X_train)
        shap.summary_plot(shap_values, self.X_train)

    def output(self, path):
        output = self.test_data[
            ["client_id", "cluster", "r_score", "m_score", "f_score"]
        ]
        output["churn_likelihood"] = self.model.predict_proba(self.X_test)[:, 1]
        output["will_churn"] = 0
        output.loc[output["churn_likelihood"] > self.threshold, "will_churn"] = 1
        output.to_csv(path)
