import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from modules.modeling import compute_mutual_information


# ======================================================
#  GLOBAL DARK TRANSPARENT PLOTLY LAYOUT
# ======================================================
def _apply_dark_style(fig, title: str):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",    # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",     # Transparent plot region
        font=dict(color="white"),
        title_font=dict(color="white", size=20),
        legend=dict(
            font=dict(color="white"),
            bgcolor="rgba(0,0,0,0)"
        )
    )
    return fig


# ======================================================
#   DATASET OVERVIEW CARDS (UNCHANGED)
# ======================================================
def render_dataset_overview_cards(df: pd.DataFrame, name: str):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f'<div class="card"><div class="card-title">Dataset</div>'
                    f'<div class="card-value">{name}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><div class="card-title">Rows</div>'
                    f'<div class="card-value">{df.shape[0]}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><div class="card-title">Columns</div>'
                    f'<div class="card-value">{df.shape[1]}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="card"><div class="card-title">Missing Values</div>'
                    f'<div class="card-value">{int(df.isna().sum().sum())}</div></div>', unsafe_allow_html=True)


# ======================================================
#   TIME SERIES (Plotly)
# ======================================================
def render_time_series(df: pd.DataFrame, target: str):
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        return

    date_col = date_cols[0]

    try:
        df_plot = df.copy()
        df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors="coerce")
        df_plot = df_plot.sort_values(by=date_col)

        if target in df_plot.columns:
            st.subheader(f"{target} over time ({date_col})")

            fig = px.line(df_plot, x=date_col, y=target)
            _apply_dark_style(fig, f"{target} Time Series")

            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


# ======================================================
#   MODEL VISUALS → Regression & Classification
# ======================================================
def render_model_visuals(model_state: dict):
    task_type = model_state["task_type"]
    X_test = model_state["X_test"]
    y_test = model_state["y_test"]
    y_pred = model_state["y_pred"]
    best_model_pipe = model_state["best_model_pipe"]

    st.subheader("Model Evaluation")

    # ---------------------------
    # Regression: Predicted vs Actual
    # ---------------------------
    if task_type == "regression":
        fig = px.scatter(
            x=y_test,
            y=y_pred,
            labels={"x": "Actual", "y": "Predicted"}
        )

        # Perfect fit line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Perfect Fit"
            )
        )

        _apply_dark_style(fig, "Predicted vs Actual")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Classification: Confusion Matrix
    # ---------------------------
    else:
        cm = confusion_matrix(y_test, y_pred)
        labels = np.unique(y_test)

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                showscale=True
            )
        )

        _apply_dark_style(fig, "Confusion Matrix")
        fig.update_xaxes(title="Predicted")
        fig.update_yaxes(title="Actual")

        st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # FEATURE IMPORTANCE (Permutation Importance)
    # ======================================================
    st.subheader("Feature Importance (Permutation)")

    try:
        perm = permutation_importance(
            best_model_pipe, X_test, y_test, n_repeats=5, random_state=42
        )

        importance_df = pd.DataFrame({
            "feature": X_test.columns,
            "importance": perm.importances_mean
        }).sort_values("importance", ascending=False)

        st.dataframe(importance_df.head(10))

        top10 = importance_df.head(10)

        fig = px.bar(
            top10,
            x="importance",
            y="feature",
            orientation="h"
        )

        _apply_dark_style(fig, "Top 10 Features (Permutation Importance)")
        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not compute feature importance: {e}")


# ======================================================
#   MUTUAL INFORMATION (Plotly)
# ======================================================
def render_mutual_information(df: pd.DataFrame, model_state: dict):
    task_type = model_state["task_type"]
    target = model_state["target"]
    best_model_name = model_state["best_model_name"]

    X_test = model_state["X_test"]
    feature_cols = list(X_test.columns)

    mi_df = compute_mutual_information(df, target, feature_cols, task_type)

    st.subheader("📎 Mutual Information Feature Importance")
    st.write("This measures how much each feature reduces uncertainty about the target.")

    st.dataframe(mi_df.head(15))

    top10 = mi_df.head(10).sort_values("mutual_information", ascending=True)

    fig = px.bar(
        top10,
        x="mutual_information",
        y="feature",
        orientation="h",
        labels={"mutual_information": "Mutual Information"}
    )

    _apply_dark_style(fig, f"Top MI Features ({best_model_name})")

    st.plotly_chart(fig, use_container_width=True)


# ======================================================
#   BASIC STAT CARDS (unchanged)
# ======================================================
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
        st.markdown(f'<div class="card"><div class="card-title">Sum</div>'
                    f'<div class="card-value">{total:,.2f}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><div class="card-title">Average</div>'
                    f'<div class="card-value">{mean:,.2f}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><div class="card-title">Min</div>'
                    f'<div class="card-value">{minimum:,.2f}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="card"><div class="card-title">Max</div>'
                    f'<div class="card-value">{maximum:,.2f}</div></div>', unsafe_allow_html=True)
