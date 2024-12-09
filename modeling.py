import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from scipy.stats import randint
import time
from typing import List, Dict
import shap


class RealEstateModel:
    """
    A class for preprocessing and training a real estate price prediction model.
    """

    def __init__(self, data_path: str, features: List[str], target: str):
        self.data_path = data_path
        self.features = features
        self.target = target
        self.df = pd.read_csv(data_path)
        self.model = None
        self.x_scaler = RobustScaler()
        self.y_scaler = RobustScaler()

    def remove_outliers(self, features_outliers_rem: List[str]) -> pd.DataFrame:
        """
        Removes outliers from specified features and the target variable.

        Args:
        - features_outliers_rem: List of feature names to check for outliers.

        Returns:
        - The DataFrame with outliers removed.
        """
        for col in features_outliers_rem + [self.target]:
            Q1 = self.df[col].quantile(0.20)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[
                (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
            ]
        return self.df

    def preprocess(self, features_outliers_rem: List[str]):
        """
        Prepares the data for training by removing outliers, scaling, and splitting.

        Args:
        - features_outliers_rem: List of feature names to check for outliers.
        """
        self.df = self.remove_outliers(features_outliers_rem)
        self.df[self.target] = np.log1p(self.df[self.target])
        X = self.df[self.features].values
        y = self.df[self.target].values.reshape(-1, 1)
        X = self.x_scaler.fit_transform(X)
        y = self.y_scaler.fit_transform(y)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomizedSearchCV:
        """
        Trains an XGBoost model using RandomizedSearchCV.

        Args:
        - X_train: Training feature set.
        - y_train: Training target set.

        Returns:
        - A fitted RandomizedSearchCV instance.
        """
        param_dist = {
            "max_depth": randint(3, 15),
            "eta": [0.01, 0.05, 0.1, 0.3, 0.5],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "alpha": [0, 0.1, 0.3, 0.5, 1.0],
            "n_estimators": [100, 200, 300, 400, 500],
            "gamma": [0, 0.01, 0.1, 0.2, 0.3],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "min_child_weight": [1, 3, 5, 7],
            "lambda": [0, 0.1, 0.5, 1.0],
        }

        model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse")
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            scoring="neg_mean_squared_error",
            n_iter=100,
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )
        random_search.fit(X_train, y_train)
        self.model = random_search.best_estimator_
        return random_search

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
        """
        Evaluates the model's performance on the test set.

        Args:
        - X_test: Test feature set.
        - y_test: Test target set.

        Returns:
        - A dictionary with RMSE, R², MAE, and MAPE scores.
        """
        y_pred = self.model.predict(X_test)
        y_pred_train = self.model.predict(X_train)

        y_pred_train_inverted = np.expm1(
            self.y_scaler.inverse_transform(y_pred_train.reshape(-1, 1))
        )
        y_pred_inverted = np.expm1(
            self.y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        )
        y_test_inverted = np.expm1(self.y_scaler.inverse_transform(y_test))
        y_train_inverted = np.expm1(self.y_scaler.inverse_transform(y_train))

        metrics_dict = {
            "RMSE_test": self.rmse(y_test_inverted, y_pred_inverted),
            "R²_test": self.rsqr(y_test_inverted, y_pred_inverted),
            "MAE_test": self.mae(y_test_inverted, y_pred_inverted),
            "MAPE_test": self.mape(y_test_inverted, y_pred_inverted),
            "sMAPE_test": self.smape(y_test_inverted, y_pred_inverted),
            "RMSE_train": self.rmse(y_train_inverted, y_pred_train_inverted),
            "R²_train": self.rsqr(y_train_inverted, y_pred_train_inverted),
            "MAE_train": self.mae(y_train_inverted, y_pred_train_inverted),
            "MAPE_train": self.mape(y_train_inverted, y_pred_train_inverted),
            "sMAPE_train": self.smape(y_train_inverted, y_pred_train_inverted),
        }
        return metrics_dict

    def rmse(self, y, pred):
        return np.sqrt(metrics.mean_squared_error(y, pred))

    def rsqr(self, y, pred):
        return metrics.r2_score(y, pred)

    def mae(self, y, pred):
        return metrics.mean_absolute_error(y, pred)

    def mape(self, y, pred):
        return np.mean(np.abs((y - pred) / y)) * 100

    def smape(self, y, pred):
        return np.mean((np.abs(y - pred)) / ((np.abs(y)) + (np.abs(pred))) / 2) * 100

    def shap_analysis(
        self, X_train: np.ndarray, X_test: np.ndarray, feature_names: List[str]
    ):
        """
        Performs SHAP analysis on the model's predictions.

        Args:
        - X_train: Training feature set.
        - X_test: Test feature set.
        - feature_names: List of feature names.
        """
        if not self.model:
            raise ValueError(
                "Model has not been trained yet. Train the model before running SHAP analysis."
            )

        explainer = shap.Explainer(self.model, X_train)
        shap_values = explainer(X_test)
        print("Generating SHAP Summary Plot...")
        shap.summary_plot(shap_values, X_test, feature_names=feature_names)

    def plot_predictions(self, y_test: np.ndarray, y_pred: np.ndarray):
        """
        Plots actual vs predicted values.

        Args:
        - y_test: Actual values from the test set.
        - y_pred: Predicted values from the test set.
        """
        y_pred_inverted = np.expm1(
            self.y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        )
        y_test_inverted = np.expm1(self.y_scaler.inverse_transform(y_test))

        plt.figure(figsize=(10, 6))
        plt.plot(y_test_inverted, label="Actual Values", color="blue", linewidth=0.7)
        plt.plot(y_pred_inverted, label="Predictions", color="orange", linewidth=0.7)
        plt.legend()
        plt.xlabel("Index")
        plt.ylabel("Price (Original Scale)")
        plt.title("Comparison of Actual Values vs Predictions")
        plt.show()
