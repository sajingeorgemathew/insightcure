import streamlit as st
from modules.modeling import train_models
from modules.theme import load_theme

load_theme()

# ---------------- AUTH CHECK ----------------
if "admin_logged_in" not in st.session_state or not st.session_state.admin_logged_in:
    st.error("Access Denied! Please login from the admin page.")
    st.stop()

# ---------------- DATASET CHECK ----------------
if "active_dataset" not in st.session_state or st.session_state.active_dataset is None:
    st.error("No dataset found! Please upload a dataset first.")
    st.stop()

df = st.session_state.active_dataset

st.title("🤖 Model Training")


# ---------------- TARGET + FEATURE SELECTION ----------------
st.subheader("Select Target & Features")

target = st.selectbox("Target Column", df.columns)
features = st.multiselect("Feature Columns", [c for c in df.columns if c != target])

# Save persistent choices
st.session_state["target_column"] = target
st.session_state["feature_columns"] = features


# ---------------- TRAIN MODEL (Unified System) ----------------
if st.button("Train Model"):
    if len(features) == 0:
        st.error("Select at least one feature.")
        st.stop()

    with st.spinner("Training models..."):
        model_state = train_models(df, target, features)

    # Persist model_state for AI Insight page
    st.session_state["model_state"] = model_state

    # Add metadata required by AI page
    dataset_name = st.session_state.get("active_dataset_name")
    st.session_state["model_state"]["dataset_name"] = (
        st.session_state.get("active_dataset_name", "Dataset")
    )
    st.session_state["model_state"]["target"] = target
    st.session_state["model_state"]["features"] = features
    st.session_state["model_state"]["model_name"] = model_state.get("best_model_name", "Unknown Model")

    # Preview predictions head (for AI Insights)
    preds = model_state.get("predictions")
    if preds is not None:
        st.session_state["model_state"]["predictions_head"] = preds.head().to_dict()

    st.success(f"Model trained successfully! Best model: {model_state['best_model_name']}")

    st.subheader("Model Performance")
    st.json(model_state["metrics"])
