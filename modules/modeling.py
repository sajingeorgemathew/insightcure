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


# =====================================================================
#  ⚡ FIXED + SAFE FEATURE ENGINEERING SUGGESTION ENGINE
# =====================================================================

def generate_feature_engineering_suggestions(df: pd.DataFrame, target_col: str):
    """
    Compute suggestions BEFORE feature selection.
    Works on the full dataset.
    Fully protected from datetime errors, object errors, etc.
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

    # ---------------------------
    # INITIAL VALIDATION
    # ---------------------------
    if target_col not in df.columns:
        return suggestions, pd.DataFrame()

    df_clean = df.dropna(subset=[target_col]).copy()
    task_type = detect_task_type(df_clean[target_col])

    # ---------------------------------------------------------
    # 1️⃣ PREPARE FEATURES — remove datetime and target column
    # ---------------------------------------------------------
    all_features = [c for c in df_clean.columns if c != target_col]

    safe_features = []
    for col in all_features:
        if np.issubdtype(df_clean[col].dtype, np.datetime64):
            continue
        safe_features.append(col)

    # If nothing left → return minimal suggestions
    if len(safe_features) == 0:
        return suggestions, pd.DataFrame()

    X = df_clean[safe_features]
    y = df_clean[target_col]

    # ---------------------------------------------------------
    # 2️⃣ SAFE ENCODING (factorize only object columns)
    # ---------------------------------------------------------
    X_encoded = pd.DataFrame(index=X.index)
    for col in safe_features:
        try:
            if X[col].dtype == "object":
                X_encoded[col], _ = X[col].factorize()
            else:
                X_encoded[col] = pd.to_numeric(X[col], errors="ignore")
        except:
            # Skip columns that cannot be encoded
            continue

    # Drop any columns that failed encoding
    X_encoded = X_encoded.select_dtypes(include=[np.number])

    # ---------------------------------------------------------
    # 3️⃣ MUTUAL INFORMATION (safe)
    # ---------------------------------------------------------
    try:
        if task_type == "classification":
            mi = mutual_info_classif(X_encoded, y, random_state=42)
        else:
            mi = mutual_info_regression(X_encoded, y, random_state=42)

        mi_df = pd.DataFrame({
            "feature": X_encoded.columns,
            "mi": mi
        }).sort_values("mi", ascending=False)

    except Exception:
        mi_df = pd.DataFrame(columns=["feature", "mi"])

    # Save top & bottom MI
    if not mi_df.empty:
        suggestions["high_mi_features"] = mi_df.head(5).to_dict(orient="records")
        suggestions["low_mi_features"] = mi_df.tail(5).to_dict(orient="records")

    # ---------------------------------------------------------
    # 4️⃣ CORRELATION DETECTION
    # ---------------------------------------------------------
    numeric_df = df_clean.select_dtypes(include=np.number)
    corr_pairs = []

    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr()

        for c1 in corr_matrix.columns:
            for c2 in corr_matrix.columns:
                if c1 < c2:
                    if abs(corr_matrix.loc[c1, c2]) > 0.80:
                        corr_pairs.append((c1, c2, corr_matrix.loc[c1, c2]))

    suggestions["high_correlation_pairs"] = corr_pairs

    # ---------------------------------------------------------
    # 5️⃣ MISSING VALUES
    # ---------------------------------------------------------
    missing = df_clean.isna().sum()

    for col, val in missing.items():
        if val > 0:
            suggestions["missing_value_warnings"].append(
                f"Column '{col}' has {val} missing values. Consider imputation."
            )

    # ---------------------------------------------------------
    # 6️⃣ HIGH CARDINALITY
    # ---------------------------------------------------------
    for col in safe_features:
        if df_clean[col].dtype == "object":
            if df_clean[col].nunique() > 30:
                suggestions["high_cardinality"].append(
                    f"Column '{col}' has very high cardinality ({df_clean[col].nunique()} unique values)."
                )

    # ---------------------------------------------------------
    # 7️⃣ ZERO VARIANCE
    # ---------------------------------------------------------
    for col in safe_features:
        if df_clean[col].nunique() <= 1:
            suggestions["zero_variance"].append(
                f"Column '{col}' has zero variance — remove it."
            )

    # ---------------------------------------------------------
    # 8️⃣ GENERAL TIPS
    # ---------------------------------------------------------
    suggestions["general_recommendations"] = [
        "Scale numeric features using StandardScaler.",
        "Extract Year / Month / Day from dates (if any).",
        "Try interaction features between highly correlated variables.",
        "Apply log-transform on skewed numeric features.",
        "Encode categorical values using One-Hot or Frequency Encoding.",
    ]

    return suggestions, mi_df



# =====================================================================
#  ORIGINAL FUNCTIONS (unchanged)
# =====================================================================

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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


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

    # remove any datetime
    dt = df_mi.select_dtypes(include=["datetime64[ns]"]).columns
    df_mi = df_mi.drop(columns=dt, errors="ignore")

    X = df_mi[feature_cols]
    y = df_mi[target_col]

    X_encoded = pd.DataFrame(index=X.index)
    for col in feature_cols:
        if X[col].dtype == "object":
            X_encoded[col], _ = X[col].factorize()
        else:
            X_encoded[col] = X[col]

    if task_type == "classification":
        mi = mutual_info_classif(X_encoded, y, random_state=42)
    else:
        mi = mutual_info_regression(X_encoded, y, random_state=42)

    mi_df = pd.DataFrame({"feature": feature_cols, "mutual_information": mi})
    return mi_df.sort_values("mutual_information", ascending=False)
