import streamlit as st
from modules.visualizations import render_dataset_overview_cards
from modules.theme import load_theme
load_theme()


if "admin_logged_in" not in st.session_state or not st.session_state.admin_logged_in:
    st.error("Access Denied! Please login from the admin page.")
    st.stop()


if "active_dataset" not in st.session_state or st.session_state.active_dataset is None:
    st.error("No dataset found! Please upload a dataset first.")
    st.stop()

df = st.session_state.active_dataset   # <-- always available now

st.title("📁 Datasets Overview")

if "datasets" not in st.session_state or len(st.session_state["datasets"]) == 0:
    st.info("No datasets loaded yet.")
else:
    for name, df in st.session_state["datasets"].items():
        render_dataset_overview_cards(df, name)
        st.dataframe(df.head())
        st.markdown("---")
