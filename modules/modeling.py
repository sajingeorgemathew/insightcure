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


# ============================================================
#  NEW FEATURE ENGINEERING SUGGESTION ENGINE  (SAFE ADDITION)
# ============================================================

def generate_feature_engineering_suggestions(df: pd.DataFrame, target_col: str, feature_cols: list):
    """
    Generates automated feature engineering suggestions based on:
    - Mutual Information
    - Correlation
    - Missing values
    - Feature cardinality
    - Zero variance features
    """

    suggestions = {
        "high_mi_features": [],
        "low_mi_features": [],
        "high_correlation_pairs": [],
        "missing_value_warnings": [],
        "high_cardinality": [],
        "zero_variance": [],
        "general_recommendations": []
    }

    df_clean = df[feature_cols + [target_col]].dropna(subset=[target_col])

    # ---------------------------
    # Detect Task Type
    # ---------------------------
    task_type = detect_task_type(df[target_col])

    # ---------------------------
    # Compute Mutual Information
    # ---------------------------
    X = df_clean[feature_cols]
    y = df_clean[target_col]

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

    mi_df = pd.DataFrame({"feature": feature_cols, "mi": mi}).sort_values("mi", ascending=False)

    # High importance features
    suggestions["high_mi_features"] = mi_df.head(5).to_dict(orient="records")

    # Low importance features
    suggestions["low_mi_features"] = mi_df.tail(5).to_dict(orient="records")

    # ---------------------------
    # Correlation Heatmap Data
    # ---------------------------
    numeric_df = df_clean.select_dtypes(include=np.number)

    corr_matrix = numeric_df.corr()
    high_corr_pairs = []

    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > 0.8:
                high_corr_pairs.append((col1, col2, corr_matrix.loc[col1, col2]))

    suggestions["high_correlation_pairs"] = high_corr_pairs

    # ---------------------------
    # Missing Values
    # ---------------------------
    missing = df_clean.isna().sum()

    for col in missing.index:
        if missing[col] > 0:
            suggestions["missing_value_warnings"].append(
                f"Column '{col}' has {missing[col]} missing values. Consider imputation."
            )

    # ---------------------------
    # Cardinality Check
    # ---------------------------
    for col in feature_cols:
        if df[col].dtype == "object":
            unique_vals = df[col].nunique()
            if unique_vals > 30:
                suggestions["high_cardinality"].append(
                    f"Column '{col}' has high cardinality ({unique_vals} unique values). Consider frequency encoding."
                )

    # ---------------------------
    # Zero Variance / Constant Features
    # ---------------------------
    for col in feature_cols:
        if df[col].nunique() <= 1:
            suggestions["zero_variance"].append(
                f"Column '{col}' has zero variance (constant). Remove it."
            )

    # ---------------------------
    # General Recommendations
    # ---------------------------
    suggestions["general_recommendations"] = [
        "Consider scaling numeric variables (StandardScaler).",
        "If date columns exist, extract Year, Month, Day, Week features.",
        "Create interaction features for highly correlated pairs.",
        "Use log-transform or Box-Cox on skewed numeric features.",
        "Encode categorical features with One-Hot or Frequency Encoding."
    ]

    return suggestions


# ============================================================
#  ORIGINAL FUNCTIONS — UNCHANGED BELOW THIS LINE
# ============================================================

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
    df_mi = df[feature_cols + [target_col]].copy()
    df_mi = df_mi.dropna(subset=[target_col])

    X = df_mi[feature_cols]
    y = df_mi[target_col]

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
