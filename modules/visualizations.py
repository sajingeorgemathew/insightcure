import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules.modeling import compute_mutual_information
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance


def render_dataset_overview_cards(df: pd.DataFrame, name: str):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="card"><div class="card-title">Dataset</div>'
                    f'<div class="card-value">{name}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><div class="card-title">Rows</div>'
                    f'<div class="card-value">{df.shape[0]}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><div class="card-title">Columns</div>'
                    f'<div class="card-value">{df.shape[1]}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="card"><div class="card-title">Missing Values</div>'
                    f'<div class="card-value">{int(df.isna().sum().sum())}</div></div>', unsafe_allow_html=True)


def render_time_series(df: pd.DataFrame, target: str):
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        return

    date_col = date_cols[0]
    try:
        df_plot = df.copy()
        df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors="coerce")
        df_plot = df_plot.sort_values(by=date_col)
        df_plot = df_plot.set_index(date_col)
        if target in df_plot.columns:
            st.subheader(f"{target} over time ({date_col})")
            st.line_chart(df_plot[target])
    except Exception:
        pass


def render_model_visuals(model_state: dict):
    task_type = model_state["task_type"]
    X_test = model_state["X_test"]
    y_test = model_state["y_test"]
    y_pred = model_state["y_pred"]
    best_model_pipe = model_state["best_model_pipe"]

    # Predictions vs Actual or Confusion Matrix
    st.subheader("Model Evaluation")

    if task_type == "regression":
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs Actual")
        st.pyplot(fig)
    else:
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    # Feature Importance
    st.subheader("Feature Importance (Permutation)")
    try:
        perm = permutation_importance(
            best_model_pipe, X_test, y_test, n_repeats=5, random_state=42
        )
        importances = perm.importances_mean
        feat_imp_df = pd.DataFrame({
            "feature": X_test.columns,
            "importance": importances
        }).sort_values("importance", ascending=False)

        st.dataframe(feat_imp_df.head(10))

        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
        top_n = feat_imp_df.head(10)
        ax_imp.barh(top_n["feature"], top_n["importance"])
        ax_imp.invert_yaxis()
        ax_imp.set_xlabel("Importance")
        ax_imp.set_title("Top 10 Features")
        st.pyplot(fig_imp)

    except Exception as e:
        st.warning(f"Could not compute feature importance: {e}")
def render_mutual_information(df: pd.DataFrame, model_state: dict):
    task_type = model_state["task_type"]
    target = model_state["target"]
    best_model_name = model_state["best_model_name"]

    # Use same feature set as model used
    X_test = model_state["X_test"]
    feature_cols = list(X_test.columns)

    mi_df = compute_mutual_information(df, target, feature_cols, task_type)

    st.subheader("📎 Mutual Information Feature Importance")
    st.write("This shows how much each feature reduces uncertainty about the target.")

    st.dataframe(mi_df.head(15))

    fig, ax = plt.subplots(figsize=(8, 4))
    top = mi_df.head(10).sort_values("mutual_information", ascending=True)
    ax.barh(top["feature"], top["mutual_information"])
    ax.set_xlabel("Mutual Information")
    ax.set_title(f"Top 10 Features by Mutual Information ({best_model_name})")
    st.pyplot(fig)
def render_basic_stats_cards(df: pd.DataFrame, column: str):
    series = df[column].dropna()
    if series.empty:
        st.warning(f"No numeric data available in column '{column}'.")
        return

    total = series.sum()
    mean = series.mean()
    minimum = series.min()
    maximum = series.max()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="card"><div class="card-title">Sum</div>'
                    f'<div class="card-value">{total:,.2f}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><div class="card-title">Average</div>'
                    f'<div class="card-value">{mean:,.2f}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><div class="card-title">Min</div>'
                    f'<div class="card-value">{minimum:,.2f}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="card"><div class="card-title">Max</div>'
                    f'<div class="card-value">{maximum:,.2f}</div></div>', unsafe_allow_html=True)

