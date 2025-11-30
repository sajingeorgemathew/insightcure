import streamlit as st
from modules.gpt_engine import generate_insight
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

st.title("🤖 AI Insight Engine")


# ---------------- MODEL CHECK ----------------
if "model_state" not in st.session_state:
    st.warning("Train a model first.")
    st.stop()

# Retrieve model_state safely
ms = st.session_state["model_state"]

# Fallback safety defaults
task_type     = ms.get("task_type", "unknown")
target        = ms.get("target", "unknown")
metrics       = ms.get("metrics", {})
dataset_name  = st.session_state.get("active_dataset_name", "Unnamed Dataset")
features      = ms.get("features", [])
model_name    = ms.get("model_name", "Unknown Model")
pred_head     = ms.get("predictions_head", None)

# Display summary above the button
st.subheader("Model Summary")
st.markdown(f"""
**Dataset:** {dataset_name}  
**Task Type:** {task_type}  
**Target:** `{target}`  
**Features:** `{features}`  
**Model:** `{model_name}`  
""")

if st.button("Generate AI Insight"):
    with st.spinner("Contacting OpenAI... interpreting your model..."):
        insight = generate_insight(
            task_type=task_type,
            target=target,
            metrics=metrics,
            dataset_name=dataset_name,
            features=features,
            model_name=model_name,
            sample_predictions=str(pred_head)
        )

    st.markdown("### AI Interpretation")
    st.write(insight)
