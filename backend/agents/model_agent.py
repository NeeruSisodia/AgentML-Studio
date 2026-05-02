import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    KFold
)
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    r2_score,
    mean_absolute_error
)
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder


class ModelAgent:
    CLASSIFICATION_MODELS = {
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=30,
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),
        "Support Vector Machine": SVC(
            kernel="linear",
            probability=True,
            random_state=42,
            max_iter=1000
        ),
    }

    REGRESSION_MODELS = {
        "Linear Regression": LinearRegression(
            n_jobs=-1
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=30,
            random_state=42
        ),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=5,
            random_state=42
        ),
        "Support Vector Regressor": SVR(
            kernel="linear"
        ),
    }

    def __init__(self):
        self.best_model = None
        self.best_name  = None
        self.task_type  = "classification"

    def _balance_classes(self, X, y):
        df = X.copy()
        df["__target__"] = y.values
        classes = df["__target__"].unique()

        if len(classes) < 2:
            return X, y

        class_counts   = df["__target__"].value_counts()
        majority_class = class_counts.index[0]
        minority_class = class_counts.index[1]

        majority = df[
            df["__target__"] == majority_class
        ]
        minority = df[
            df["__target__"] == minority_class
        ]

        minority_upsampled = resample(
            minority,
            replace=True,
            n_samples=len(majority),
            random_state=42
        )

        df_balanced = pd.concat([
            majority,
            minority_upsampled
        ])
        df_balanced = df_balanced.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

        X_balanced = df_balanced.drop(
            columns=["__target__"]
        )
        y_balanced = df_balanced["__target__"]
        return X_balanced, y_balanced

    def train_and_select(self, X, y):
        if X is None or y is None:
            raise ValueError(
                "X and y cannot be None. "
                "Run feature engineering first."
            )

        X = X.fillna(0)

        n_unique      = y.nunique()
        is_regression = (
            n_unique > 20 and
            y.dtype in [
                "float64", "float32", "int64"
            ]
        )

        if is_regression:
            self.task_type = "regression"
            return self._train_regression(X, y)
        else:
            self.task_type = "classification"
            return self._train_classification(X, y)

    def _train_classification(self, X, y):
        if y.dtype == "object":
            le = LabelEncoder()
            y  = pd.Series(
                le.fit_transform(y),
                index=y.index
            )

        if y.value_counts(normalize=True).min() < 0.2:
            X, y = self._balance_classes(X, y)

        cv = StratifiedKFold(
            n_splits=2,
            shuffle=True,
            random_state=42
        )

        best_score    = 0
        best_name     = None
        best_model    = None
        model_results = []

        for name, model in (
            self.CLASSIFICATION_MODELS.items()
        ):
            try:
                scores = cross_val_score(
                    model, X, y,
                    cv=cv,
                    scoring="accuracy"
                )
                print(
                    f"  {name}: "
                    f"accuracy={scores.mean():.3f}"
                )
                model_results.append({
                    "model":    name,
                    "accuracy": round(
                        float(scores.mean()), 4
                    ),
                    "std":      round(
                        float(scores.std()), 4
                    )
                })
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_name  = name
                    best_model = model
            except Exception as e:
                print(f"Model {name} failed: {e}")
                continue

        if best_model is None:
            best_name  = "Random Forest"
            best_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            best_score = 0.0

        best_model.fit(X, y)
        y_pred = best_model.predict(X)

        is_binary = len(y.unique()) == 2
        proba = (
            best_model.predict_proba(X)[:, 1]
            if (
                hasattr(best_model, "predict_proba")
                and is_binary
            )
            else None
        )

        self.best_model = best_model
        self.best_name  = best_name

        try:
            auc = round(float(
                roc_auc_score(y, proba)
            ), 4) if proba is not None else 0.0
        except Exception:
            auc = 0.0

        return {
            "model_name":    best_name,
            "task":          "classification",
            "accuracy":      round(
                float(best_score), 4
            ),
            "f1":            round(float(
                f1_score(
                    y, y_pred,
                    average="weighted"
                )
            ), 4),
            "auc":           auc,
            "model_results": model_results
        }

    def _train_regression(self, X, y):
        cv = KFold(
            n_splits=2,
            shuffle=True,
            random_state=42
        )

        best_score    = -999
        best_name     = None
        best_model    = None
        model_results = []

        for name, model in (
            self.REGRESSION_MODELS.items()
        ):
            try:
                scores = cross_val_score(
                    model, X, y,
                    cv=cv,
                    scoring="r2"
                )
                print(
                    f"  {name}: "
                    f"r2={scores.mean():.3f}"
                )
                model_results.append({
                    "model":    name,
                    "accuracy": round(
                        float(scores.mean()), 4
                    ),
                    "std":      round(
                        float(scores.std()), 4
                    )
                })
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_name  = name
                    best_model = model
            except Exception as e:
                print(f"Model {name} failed: {e}")
                continue

        if best_model is None:
            best_name  = "Random Forest Regressor"
            best_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
            best_score = 0.0

        best_model.fit(X, y)
        y_pred = best_model.predict(X)

        r2  = round(float(r2_score(y, y_pred)), 4)
        mae = round(float(
            mean_absolute_error(y, y_pred)
        ), 4)

        self.best_model = best_model
        self.best_name  = best_name

        return {
            "model_name":    best_name,
            "task":          "regression",
            "accuracy":      max(0.0, r2),
            "r2_score":      r2,
            "mae":           mae,
            "f1":            0.0,
            "auc":           0.0,
            "model_results": model_results
        }

    def save_model(self):
        os.makedirs("models", exist_ok=True)
        joblib.dump(
            self.best_model,
            "models/latest.pkl"
        )