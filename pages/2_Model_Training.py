import streamlit as st
from modules.modeling import (
    train_models,
    generate_feature_engineering_suggestions
)
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

# ===============================================================
# 1️⃣ TARGET SELECTION (ALWAYS FIRST STEP)
# ===============================================================
st.subheader("Select Target Column")

target = st.selectbox("Target Column", df.columns)
st.session_state["target_column"] = target

# ===============================================================
# 2️⃣ FEATURE ENGINEERING SUGGESTIONS (BEFORE feature selection)
# ===============================================================
st.markdown("### 🧠 Automated Feature Engineering Suggestions")

if st.button("Generate Feature Engineering Suggestions"):
    with st.spinner("Analyzing dataset…"):
        suggestions, mi_df = generate_feature_engineering_suggestions(df, target)

    st.success("Feature Engineering Suggestions Ready")

    # -------- Show MI Table --------
    st.write("### 🔍 Mutual Information Ranking")
    if not mi_df.empty:
        st.dataframe(mi_df)
    else:
        st.info("MI could not be computed for this dataset.")

    # -------- Display Suggestion Groups --------
    with st.expander("🔥 High Importance Features"):
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
            st.write("No missing value issues found.")

    with st.expander("📦 High Cardinality Categorical Columns"):
        if suggestions["high_cardinality"]:
            st.write(suggestions["high_cardinality"])
        else:
            st.write("No high-cardinality columns detected.")

    with st.expander("🚫 Zero Variance Columns"):
        if suggestions["zero_variance"]:
            st.write(suggestions["zero_variance"])
        else:
            st.write("No zero-variance columns detected.")

    with st.expander("💡 General Recommendations"):
        st.write(suggestions["general_recommendations"])

# ===============================================================
# 3️⃣ FEATURE SELECTION (AFTER suggestions)
# ===============================================================
st.subheader("Select Features for Model Training")

default_features = [c for c in df.columns if c != target]
features = st.multiselect("Feature Columns", default_features)

st.session_state["feature_columns"] = features

# ===============================================================
# 4️⃣ TRAIN MODEL (UNCHANGED LOGIC)
# ===============================================================
st.subheader("Train the Model")

if st.button("Train Model"):
    if len(features) == 0:
        st.error("Select at least one feature before training.")
        st.stop()

    with st.spinner("Training models..."):
        model_state = train_models(df, target, features)

    # Persist model state
    st.session_state["model_state"] = model_state
    st.session_state["model_state"]["dataset_name"] = st.session_state.get(
        "active_dataset_name", "Dataset"
    )
    st.session_state["model_state"]["target"] = target
    st.session_state["model_state"]["features"] = features
    st.session_state["model_state"]["model_name"] = model_state.get(
        "best_model_name", "Unknown Model"
    )

    st.success(f"Model trained successfully! Best model: {model_state['best_model_name']}")

    st.subheader("Model Performance")
    st.json(model_state["metrics"])
