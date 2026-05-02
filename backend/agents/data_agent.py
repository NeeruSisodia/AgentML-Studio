import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)


class DataAgent:
    def __init__(self):
        self.df = None
        self.X  = None
        self.y  = None

    def ingest(self, path):
        df = pd.read_csv(path)
        report = {
            "duplicates": int(
                df.duplicated().sum()
            ),
            "imputed": 0
        }

        df.drop_duplicates(inplace=True)

        # Number columns filled with median value
        for col in df.columns:
            if df[col].dtype == "object":
                fill_val = (
                    df[col].mode()[0]
                    if not df[col].mode().empty
                    else "unknown"
                )
                df[col] = df[col].fillna(fill_val)
            else:
                df[col] = df[col].fillna(
                    df[col].median()
                )

        #  Encode text columns to numbers
        for col in df.select_dtypes(
            include="object"
        ).columns:
            df[col] = LabelEncoder().fit_transform(
                df[col].astype(str)
            )

        #  Fill any remaining missing values
        df = df.fillna(0)
        self.df = df
        return df, report

    def analyse(self):
        target   = self.df.columns[-1]
        n_unique = self.df[target].nunique()

        # Determine task type based on target variable
        if (
            n_unique <= 20 or
            self.df[target].dtype == "object"
        ):
            task = "classification"
        else:
            task = "regression"

        if task == "classification":
            balance = float(
                self.df[target].value_counts(
                    normalize=True
                ).min()
            )
        else:
            balance = 0.5

        return {
            "balance":   balance,
            "n_classes": int(n_unique),
            "null_pct":  float(
                self.df.isnull().mean().mean()
            ),
            "task":      task
        }

    def engineer_features(self):
        try:
            target = self.df.columns[-1]
            X = self.df.drop(columns=[target])
            y = self.df[target]

            #  Fill any missing values in features
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)

            #  Encode any remaining text columns
            for col in X.columns:
                if X[col].dtype == "object":
                    le = LabelEncoder()
                    X[col] = le.fit_transform(
                        X[col].astype(str)
                    )

            if X.shape[1] == 0:
                raise ValueError(
                    "No features available"
                )

            n_unique = y.nunique()
            is_regression = (
                n_unique > 20 and
                y.dtype in [
                    "float64",
                    "float32",
                    "int64"
                ]
            )

            try:
                if is_regression:
                    #  use f_regression for regression tasks
                    from sklearn.feature_selection \
                        import f_regression
                    scores, _ = f_regression(X, y)
                    scores    = np.nan_to_num(scores)
                    if scores.max() > 0:
                        cols = X.columns[
                            scores > np.percentile(
                                scores, 25
                            )
                        ].tolist()
                    else:
                        cols = X.columns.tolist()
                else:
                    #  use mutual_info for classification tasks
                    from sklearn.feature_selection \
                        import mutual_info_classif
                    if y.dtype == "object":
                        le        = LabelEncoder()
                        y_encoded = le.fit_transform(y)
                    else:
                        y_encoded = y

                    mi = mutual_info_classif(
                        X, y_encoded,
                        random_state=42
                    )
                    cols = X.columns[
                        mi > np.percentile(mi, 25)
                    ].tolist()

            except Exception as e:
                print(f"Feature selection error: {e}")
                cols = X.columns.tolist()

            #  make sure we always have at least one feature
            if len(cols) == 0:
                cols = X.columns.tolist()

            X = X[cols]

            try:
                #  Calculate feature importance
                if is_regression:
                    rf = RandomForestRegressor(
                        n_estimators=30,
                        random_state=42
                    )
                else:
                    rf = RandomForestClassifier(
                        n_estimators=30,
                        random_state=42
                    )
                rf.fit(X, y)
                imp = dict(zip(
                    cols,
                    rf.feature_importances_.tolist()
                ))
            except Exception as e:
                print(f"Importance error: {e}")
                imp = {
                    col: round(1.0 / len(cols), 4)
                    for col in cols
                }

            self.X = X
            self.y = y

            print(
                f"Features ready: "
                f"{X.shape[1]} cols, "
                f"{X.shape[0]} rows"
            )

            return cols, imp

        except Exception as e:
            #  Emergency fallback if something fails
            print(f"engineer_features error: {e}")
            target = self.df.columns[-1]
            X = self.df.drop(
                columns=[target]
            ).fillna(0)
            y      = self.df[target]
            self.X = X
            self.y = y
            imp = {
                col: round(1.0 / len(X.columns), 4)
                for col in X.columns
            }
            print(
                f"Emergency fix: "
                f"{X.shape[1]} features"
            )
            return X.columns.tolist(), imp