import streamlit as st
import pandas as pd
from modules.preprocessing import enrich_date_features
from modules.theme import load_theme

# Load global theme
load_theme()

# -------- AUTH CHECK --------
if "admin_logged_in" not in st.session_state or not st.session_state.admin_logged_in:
    st.error("Access Denied! Please login from the admin page.")
    st.stop()

st.title("📁 Upload Data")

# -------- GLOBAL DATASET STORAGE (persistent across pages) --------
if "active_dataset_name" not in st.session_state:
    st.session_state["active_dataset_name"] = None

if "active_dataset" not in st.session_state:
    st.session_state["active_dataset"] = None

# -------- UI ELEMENTS --------
dataset_name = st.text_input("Dataset Name", value="Dataset_1")
dataset_name = dataset_name.strip().replace(" ", "_")



uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df, date_cols = enrich_date_features(df)

    st.success(f"File '{dataset_name}' uploaded successfully!")
    st.dataframe(df.head())

    # -------- SAVE TO SESSION --------
    if st.button("Save Dataset"):
        st.session_state["active_dataset"] = df
        st.session_state["active_dataset_name"] = dataset_name
        # NEW: also save inside datasets dict for visualization page
    if "datasets" not in st.session_state:
        st.session_state["datasets"] = {}

    st.session_state["datasets"][dataset_name] = df

    st.success(f"Dataset '{dataset_name}' is now active and persistent across all pages.")