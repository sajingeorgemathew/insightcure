import streamlit as st
from modules.visualizations import (
    render_model_visuals,
    render_mutual_information,
    render_basic_stats_cards,
)
from modules.descriptive import (
    plot_histogram,
    plot_boxplot,
    plot_scatter,
    plot_corr_heatmap,
    plot_pairplot,
    plot_bar,
)
from modules.theme import load_theme

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

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    # ---------- HISTOGRAM ----------
    st.markdown("### Histogram")
    hist_col = st.selectbox("Select a numeric column", numeric_cols, key="hist_col")
    if st.button("Plot Histogram"):
        plot_histogram(df, hist_col)

    st.markdown("---")

    # ---------- BOXPLOT ----------
    st.markdown("### Boxplot")
    box_col = st.selectbox("Select a column", numeric_cols, key="box_col")
    if st.button("Plot Boxplot"):
        plot_boxplot(df, box_col)

    st.markdown("---")

    # ---------- SCATTER ----------
    st.markdown("### Scatter Plot")
    scatter_x = st.selectbox("X-axis", numeric_cols, key="scatter_x")
    scatter_y = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
    if st.button("Plot Scatter"):
        plot_scatter(df, scatter_x, scatter_y)

    st.markdown("---")

    # ---------- CORRELATION HEATMAP ----------
    st.markdown("### Correlation Heatmap")
    heat_cols = st.multiselect(
        "Select numeric columns",
        numeric_cols,
        default=numeric_cols[:5]
    )
    if st.button("Plot Heatmap"):
        plot_corr_heatmap(df, heat_cols)

    st.markdown("---")

    # ---------- PAIRPLOT ----------
    st.markdown("### Pairplot")
    pair_cols = st.multiselect(
        "Select up to 5 columns",
        numeric_cols,
        default=numeric_cols[:3]
    )
    if 0 < len(pair_cols) <= 5:
        if st.button("Generate Pairplot"):
            plot_pairplot(df, pair_cols)


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
