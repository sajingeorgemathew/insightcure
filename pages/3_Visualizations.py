import streamlit as st
from modules.visualizations import (
    render_model_visuals,
    render_time_series,
    render_mutual_information,
)
from modules.theme import load_theme
load_theme()


if "admin_logged_in" not in st.session_state or not st.session_state.admin_logged_in:
    st.error("Access Denied! Please login from the admin page.")
    st.stop()


if "active_dataset" not in st.session_state or st.session_state.active_dataset is None:
    st.error("No dataset found! Please upload a dataset first.")
    st.stop()

dataset_key = st.session_state.get("active_dataset_name")

if "datasets" not in st.session_state or dataset_key not in st.session_state["datasets"]:
    st.error("Dataset not available in memory. Please re-upload or retrain.")
    st.stop()

df = st.session_state["datasets"][dataset_key]

st.title("Visualizations")

if "model_state" not in st.session_state:
    st.warning("Train a model first in the Model Training page.")
else:
    ms = st.session_state["model_state"]
    dataset_name = ms.get("dataset_name")

    # Validate dataset existence
    if (
        "datasets" not in st.session_state
        or dataset_name not in st.session_state["datasets"]
    ):
        st.error("Dataset not found. Please re-upload or retrain the model.")
        st.stop()

    df = st.session_state["datasets"][dataset_name]

    tab1, tab2, tab3 = st.tabs(["Time Series", "Model Performance", "Mutual Information"])

    with tab1:
        render_time_series(df, ms["target"])
    with tab2:
        render_model_visuals(ms)
    with tab3:
        render_mutual_information(df, ms)
from modules.visualizations import render_basic_stats_cards

# After tabs
st.markdown("---")
st.subheader("Quick Stats")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if numeric_cols:
    stat_col = st.selectbox("Select a numeric column for stats", numeric_cols)
    if st.button("Show Stats Cards"):
        render_basic_stats_cards(df, stat_col)
else:
    st.info("No numeric columns detected for stats.")

