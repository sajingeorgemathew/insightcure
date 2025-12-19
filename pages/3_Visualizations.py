import streamlit as st
from modules.visualizations import (
    render_model_visuals,
    render_mutual_information,
    render_basic_stats_cards,
)
from modules.descriptive import (
    plot_histogram,
    plot_corr_heatmap,
    plot_pie,
    plot_column_chart,
    plot_category_numeric_heatmap,
)
from modules.gpt_engine import generate_insight
from modules.theme import load_theme
import pandas as pd
import numpy as np

# Load theme
load_theme()

# ---------------- AUTH CHECK ----------------
if "admin_logged_in" not in st.session_state or not st.session_state.admin_logged_in:
    st.error("Access Denied! Please login from the admin page.")
    st.stop()

# ---------------- DATASET CHECK ----------------
if "active_dataset" not in st.session_state or st.session_state.active_dataset is None:
    st.error("No dataset found! Please upload a dataset first.")
    st.stop()

dataset_key = st.session_state.get("active_dataset_name")

if "datasets" not in st.session_state or dataset_key not in st.session_state["datasets"]:
    st.error("Dataset not available in memory. Please re-upload or retrain.")
    st.stop()

df = st.session_state["datasets"][dataset_key]

st.title("Visualizations Dashboard")

# ---------------- MODEL CHECK ----------------
if "model_state" not in st.session_state:
    st.warning("Train a model first in the Model Training page.")
    st.stop()

ms = st.session_state["model_state"]
dataset_name = ms.get("dataset_name")

# Validate correct dataset
if dataset_name not in st.session_state["datasets"]:
    st.error("Dataset mismatch. Please re-upload or retrain.")
    st.stop()

df = st.session_state["datasets"][dataset_name]

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs([
    "Model Performance",
    "Mutual Information",
    "Descriptive Plots"
])

# ======================================================
# TAB 1 — MODEL PERFORMANCE (Plotly)
# ======================================================
with tab1:
    st.markdown("### Model Performance Overview")
    render_model_visuals(ms)


# ======================================================
# TAB 2 — MUTUAL INFORMATION (Plotly)
# ======================================================
with tab2:
    st.markdown("### Mutual Information Analysis")
    render_mutual_information(df, ms)


# ======================================================
# TAB 3 — DESCRIPTIVE PLOTS (Plotly)
# ======================================================
with tab3:
    st.subheader("Descriptive Analytics")
    st.caption(
        "Interpret buttons use summarized statistics (not full raw data or images) to generate "
        "a client-friendly explanation."
    )

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    def render_insight_button(chart_label, context_payload, button_key):
        if st.button(f"Interpret {chart_label}", key=button_key):
            with st.spinner("Generating interpretation..."):
                insight = generate_insight(
                    task_type=ms.get("task_type", "unknown"),
                    target=ms.get("target", "unknown"),
                    metrics=ms.get("metrics", {}),
                    dataset_name=ms.get("dataset_name", "Dataset"),
                    chart=chart_label,
                    context=context_payload,
                )
            st.markdown("**AI Interpretation**")
            st.write(insight)

    # ---------- HISTOGRAM ----------
    st.markdown("### Histogram")
    if not numeric_cols:
        st.info("No numeric columns available for histogram.")
    else:
        hist_col = st.selectbox("Select a numeric column", numeric_cols, key="hist_col")
        if st.button("Plot Histogram", key="plot_histogram"):
            st.session_state["hist_ready"] = True
        if st.session_state.get("hist_ready"):
            plot_histogram(df, hist_col)
            hist_series = df[hist_col].dropna()
            hist_context = {
                "column": hist_col,
                "count": int(hist_series.count()),
                "mean": float(hist_series.mean()) if not hist_series.empty else None,
                "median": float(hist_series.median()) if not hist_series.empty else None,
                "min": float(hist_series.min()) if not hist_series.empty else None,
                "max": float(hist_series.max()) if not hist_series.empty else None,
                "std": float(hist_series.std()) if not hist_series.empty else None,
            }
            render_insight_button("Histogram", hist_context, "interpret_histogram")

    st.markdown("---")

    # ---------- PIE CHART ----------
    st.markdown("### Pie Chart")
    if not categorical_cols:
        st.info("No categorical columns available for pie charts.")
    else:
        pie_col = st.selectbox("Select a categorical column", categorical_cols, key="pie_col")
        if st.button("Plot Pie Chart", key="plot_pie"):
            st.session_state["pie_ready"] = True
        if st.session_state.get("pie_ready"):
            plot_pie(df, pie_col)
            counts = df[pie_col].value_counts(dropna=False)
            top_counts = counts.head(5)
            pie_context = {
                "column": pie_col,
                "top_categories": top_counts.to_dict(),
                "unique_count": int(counts.shape[0]),
            }
            render_insight_button("Pie Chart", pie_context, "interpret_pie")

    st.markdown("---")

    # ---------- COLUMN CHART ----------
    st.markdown("### Column Chart (Category vs Numeric)")
    if not categorical_cols or not numeric_cols:
        st.info("Need both categorical and numeric columns for column charts.")
    else:
        column_cat = st.selectbox("Category column", categorical_cols, key="column_cat")
        column_num = st.selectbox("Numeric column", numeric_cols, key="column_num")
        column_agg = st.selectbox("Aggregation", ["mean", "sum", "median", "count"], key="column_agg")
        if st.button("Plot Column Chart", key="plot_column"):
            st.session_state["column_ready"] = True
        if st.session_state.get("column_ready"):
            plot_column_chart(df, column_cat, column_num, column_agg)
            grouped = df.groupby(column_cat)[column_num].agg(column_agg).sort_values(ascending=False)
            column_context = {
                "category_column": column_cat,
                "numeric_column": column_num,
                "aggregation": column_agg,
                "top_categories": grouped.head(5).to_dict(),
            }
            render_insight_button("Column Chart", column_context, "interpret_column")

    st.markdown("---")

    # ---------- CORRELATION HEATMAP ----------
    st.markdown("### Correlation Heatmap")
    if not numeric_cols:
        st.info("No numeric columns available for correlation heatmaps.")
    else:
        heat_cols = st.multiselect(
            "Select numeric columns",
            numeric_cols,
            default=numeric_cols[:5]
        )
        if st.button("Plot Heatmap", key="plot_corr_heatmap"):
            st.session_state["corr_heat_ready"] = True
        if st.session_state.get("corr_heat_ready"):
            plot_corr_heatmap(df, heat_cols)
            if len(heat_cols) > 1:
                corr = df[heat_cols].corr()
                corr_context = {
                    "columns": heat_cols,
                    "correlations": corr.round(3).to_dict(),
                }
                render_insight_button("Correlation Heatmap", corr_context, "interpret_corr_heatmap")

    st.markdown("---")

    # ---------- CATEGORY VS NUMERIC HEATMAP ----------
    st.markdown("### Category vs Numeric Heatmap")
    if not categorical_cols or not numeric_cols:
        st.info("Need both categorical and numeric columns for category heatmaps.")
    else:
        cat_heat_cat = st.selectbox("Category column", categorical_cols, key="cat_heat_cat")
        cat_heat_num = st.selectbox("Numeric column", numeric_cols, key="cat_heat_num")
        cat_heat_agg = st.selectbox(
            "Aggregation",
            ["mean", "sum", "median", "count"],
            key="cat_heat_agg"
        )
        if st.button("Plot Category Heatmap", key="plot_cat_heatmap"):
            st.session_state["cat_heat_ready"] = True
        if st.session_state.get("cat_heat_ready"):
            plot_category_numeric_heatmap(df, cat_heat_cat, cat_heat_num, cat_heat_agg)
            grouped = df.groupby(cat_heat_cat)[cat_heat_num].agg(cat_heat_agg).sort_values(ascending=False)
            cat_heat_context = {
                "category_column": cat_heat_cat,
                "numeric_column": cat_heat_num,
                "aggregation": cat_heat_agg,
                "top_categories": grouped.head(5).to_dict(),
            }
            render_insight_button(
                "Category vs Numeric Heatmap",
                cat_heat_context,
                "interpret_cat_heatmap"
            )

    st.markdown("---")

    # ---------- CORRELATION STRENGTH TABLE ----------
    st.markdown("### Correlation Strength Table")
    if not numeric_cols:
        st.info("No numeric columns available for correlation tables.")
    else:
        corr_cols = st.multiselect(
            "Select numeric columns for correlation table",
            numeric_cols,
            default=numeric_cols[:5],
            key="corr_table_cols"
        )
        if st.button("Generate Correlation Table", key="plot_corr_table"):
            st.session_state["corr_table_ready"] = True
        if st.session_state.get("corr_table_ready"):
            if len(corr_cols) < 2:
                st.warning("Select at least two numeric columns.")
            else:
                corr = df[corr_cols].corr()
                corr_pairs = (
                    corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
                    .stack()
                    .reset_index()
                )
                corr_pairs.columns = ["Feature A", "Feature B", "Correlation"]
                corr_pairs = corr_pairs.sort_values("Correlation", ascending=False)
                st.dataframe(corr_pairs)
                corr_table_context = {
                    "columns": corr_cols,
                    "top_correlations": corr_pairs.head(5).to_dict(orient="records"),
                }
                render_insight_button(
                    "Correlation Strength Table",
                    corr_table_context,
                    "interpret_corr_table"
                )


# ======================================================
# QUICK STATS — MOVED BELOW TABS
# ======================================================
st.markdown("---")
st.subheader(" Quick Stats Summary")

numeric_cols = df.select_dtypes(include="number").columns.tolist()

if numeric_cols:
    stat_col = st.selectbox("Choose a numeric column", numeric_cols)
    if st.button("Show Column Statistics"):
        render_basic_stats_cards(df, stat_col)
else:
    st.info("No numeric columns found in this dataset.")
