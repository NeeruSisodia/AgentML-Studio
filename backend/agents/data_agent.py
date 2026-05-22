import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class DataAgent:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        

    def ingest(self, path):
        df = pd.read_csv(path)
        n_before = len(df)
        df.drop_duplicates(inplace=True)
        dupes = n_before - len(df)

        # median for numerics avoids skew from outliers, more categoricals
        for col in df.columns:
            if df[col].dtype == "object":
                m = df[col].mode()
                df[col].fillna(m[0] if len(m) else "unknown", inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

        # encode strings yes label encoding isn't ideal for everything
        # OHE would be better for high cardinality but this is fast
        cats = df.select_dtypes(include="object").columns
        for col in cats:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        self.df = df
        return df, {"duplicates": dupes, "imputed": 0}

    def analyse(self):
        # just assume last col is target, works 90% of the time
        target = self.df.columns[-1]
        n_unique = self.df[target].nunique()

        # rough heuristic, 20 classes = classification
        if n_unique <= 20 or self.df[target].dtype == "object":
            task = "classification"
            balance = float(self.df[target].value_counts(normalize=True).min())
        else:
            task = "regression"
            balance = 0.5

        return {
            "task": task,
            "n_classes": int(n_unique),
            "class_balance": balance,   
            "null_pct": float(self.df.isnull().mean().mean()),
        }

    def engineer_features(self):
        target = self.df.columns[-1]

        try:
            X = self.df.drop(columns=[target]).fillna(0)
            y = self.df[target]

            # inf values can sneak in from log transforms upstream
            X.replace([np.inf, -np.inf], 0, inplace=True)

            # re-encode just in case, got burned skipping this once
            for col in X.select_dtypes(include="object").columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

            if X.shape[1] == 0:
                raise ValueError("dropped everything, nothing left")

            # same threshold as analyse()  bit duplicated but meh
            is_reg = y.nunique() > 20 and y.dtype in ["float64", "float32", "int64"]

            kept_cols = self._pick_features(X, y, is_reg)
            X = X[kept_cols]
            importances = self._feat_importance(X, y, kept_cols, is_reg)

        except Exception as e:
            # yes broad except, this is best-effort 
            print(f"feature step blew up, falling back to all cols: {e}")
            X = self.df.drop(columns=[target]).fillna(0)
            y = self.df[target]
            kept_cols = X.columns.tolist()
            importances = {c: round(1 / len(kept_cols), 4) for c in kept_cols}

        self.X = X
        self.y = y
        print(f"done: {X.shape[1]} features, {X.shape[0]} rows")
        return kept_cols, importances

    # bottom 25% of features by score get dropped
    def _pick_features(self, X, y, is_reg):
        try:
            if is_reg:
                from sklearn.feature_selection import f_regression
                scores, _ = f_regression(X, y)
                scores = np.nan_to_num(scores)
                if scores.max() == 0:
                    return X.columns.tolist()
                return X.columns[scores > np.percentile(scores, 25)].tolist()
            else:
                from sklearn.feature_selection import mutual_info_classif
                y2 = LabelEncoder().fit_transform(y) if y.dtype == "object" else y
                mi = mutual_info_classif(X, y2, random_state=42)
                cols = X.columns[mi > np.percentile(mi, 25)].tolist()
                return cols or X.columns.tolist()
        except Exception as e:
            print(f"selection failed: {e}")
            return X.columns.tolist()

    def _feat_importance(self, X, y, cols, is_reg):
        try:
            # 30 trees is plenty for importance ranking, not trying to win kaggle
            clf = RandomForestRegressor if is_reg else RandomForestClassifier
            m = clf(n_estimators=30, random_state=42)
            m.fit(X, y)
            return dict(zip(cols, map(float, m.feature_importances_)))
        except Exception as e:
            print(f"importance failed: {e}")
            return {c: round(1 / len(cols), 4) for c in cols}
