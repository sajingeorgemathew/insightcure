import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, r2_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


def detect_task_type(y: pd.Series) -> str:
    if y.dtype == "object" or y.nunique() < 20:
        return "classification"
    return "regression"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def get_models(task_type: str):
    if task_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(eval_metric="logloss"),
        }
    else:
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "XGBoost": XGBRegressor(),
        }


def train_models(df: pd.DataFrame, target_col: str, feature_cols: list):
    df_model = df[feature_cols + [target_col]].copy()

    # Drop datetime and target NaNs
    datetime_cols = df_model.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    df_model = df_model.drop(columns=datetime_cols)
    df_model = df_model.dropna(subset=[target_col])

    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    task_type = detect_task_type(y)
    preprocessor = build_preprocessor(X)
    models = get_models(task_type)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_model_name = None
    best_model_pipe = None
    best_score = -999
    metrics = {}
    preds_store = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        if task_type == "classification":
            score = f1_score(y_test, preds, average="weighted")
            metrics[name] = {"F1 Score (weighted)": float(score)}
        else:
            score = r2_score(y_test, preds)
            metrics[name] = {"R2 Score": float(score)}

        preds_store[name] = preds

        if score > best_score:
            best_score = score
            best_model_name = name
            best_model_pipe = pipe

    # Recompute preds for best model
    best_preds = preds_store[best_model_name]

    return {
        "task_type": task_type,
        "metrics": metrics,
        "best_model_name": best_model_name,
        "best_model_pipe": best_model_pipe,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": best_preds,
    }
def compute_mutual_information(df: pd.DataFrame, target_col: str, feature_cols: list, task_type: str):
    """
    Compute mutual information between features and target.
    Returns a DataFrame sorted by MI descending.
    """
    df_mi = df[feature_cols + [target_col]].copy()
    df_mi = df_mi.dropna(subset=[target_col])

    X = df_mi[feature_cols]
    y = df_mi[target_col]

    # Basic encoding for non-numeric features
    X_encoded = pd.DataFrame(index=X.index)
    for col in feature_cols:
        if X[col].dtype == "object":
            X_encoded[col], _ = X[col].factorize()
        else:
            X_encoded[col] = X[col]

    if task_type == "classification":
        mi = mutual_info_classif(X_encoded, y, discrete_features="auto", random_state=42)
    else:
        mi = mutual_info_regression(X_encoded, y, random_state=42)

    mi_df = pd.DataFrame({"feature": feature_cols, "mutual_information": mi})
    mi_df = mi_df.sort_values("mutual_information", ascending=False)
    return mi_df
