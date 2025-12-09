import streamlit as st
from modules.modeling import train_models, generate_feature_engineering_suggestions
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

# ===============================================================
# 🔍 NEW: FEATURE ENGINEERING SUGGESTION BUTTON
# ===============================================================
st.markdown("### 🧠 Feature Engineering Suggestions")

if st.button("Generate Feature Engineering Suggestions"):
    if len(features) == 0:
        st.error("Please select at least one feature to analyze.")
        st.stop()

    with st.spinner("Analyzing features..."):
        suggestions = generate_feature_engineering_suggestions(df, target, features)

    st.success("Feature Engineering Suggestions Ready")

    # -------- Display Suggestions in Expanders --------
    with st.expander("🔥 High Importance Features (Mutual Information)"):
        st.write(suggestions["high_mi_features"])

    with st.expander("🧊 Low Importance Features"):
        st.write(suggestions["low_mi_features"])

    with st.expander("🔗 Highly Correlated Feature Pairs (> 0.8)"):
        if suggestions["high_correlation_pairs"]:
            st.write(suggestions["high_correlation_pairs"])
        else:
            st.write("No strong correlations detected.")

    with st.expander("⚠ Missing Value Warnings"):
        if suggestions["missing_value_warnings"]:
            st.write(suggestions["missing_value_warnings"])
        else:
            st.write("No missing value issues detected.")

    with st.expander("📦 High Cardinality Categorical Columns"):
        if suggestions["high_cardinality"]:
            st.write(suggestions["high_cardinality"])
        else:
            st.write("No high-cardinality issues found.")

    with st.expander("🚫 Zero Variance Columns"):
        if suggestions["zero_variance"]:
            st.write(suggestions["zero_variance"])
        else:
            st.write("No zero-variance columns detected.")

    with st.expander("💡 General Recommendations"):
        st.write(suggestions["general_recommendations"])


# ===============================================================
# TRAIN MODEL (UNMODIFIED LOGIC)
# ===============================================================
st.subheader("Train the Model")

if st.button("Train Model"):
    if len(features) == 0:
        st.error("Select at least one feature.")
        st.stop()

    with st.spinner("Training models..."):
        model_state = train_models(df, target, features)

    # Save in session
    st.session_state["model_state"] = model_state
    st.session_state["model_state"]["dataset_name"] = st.session_state.get("active_dataset_name", "Dataset")
    st.session_state["model_state"]["target"] = target
    st.session_state["model_state"]["features"] = features
    st.session_state["model_state"]["model_name"] = model_state.get("best_model_name", "Unknown Model")

    st.success(f"Model trained successfully! Best model: {model_state['best_model_name']}")

    st.subheader("Model Performance")
    st.json(model_state["metrics"])
