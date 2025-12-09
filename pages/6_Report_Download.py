import io
import datetime

import streamlit as st
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from modules.theme import load_theme
from modules.modeling import generate_feature_engineering_suggestions

# ---------------- THEME ----------------
load_theme()

# ---------------- AUTH CHECK ----------------
if "admin_logged_in" not in st.session_state or not st.session_state.admin_logged_in:
    st.error("Access Denied! Please login from the admin page.")
    st.stop()

# ---------------- DATASET CHECK ----------------
if "active_dataset" not in st.session_state or st.session_state.active_dataset is None:
    st.error("No dataset found! Please upload a dataset and train a model first.")
    st.stop()

if "model_state" not in st.session_state:
    st.error("No trained model found. Please train a model first.")
    st.stop()

df = st.session_state.active_dataset
model_state = st.session_state["model_state"]

dataset_name = st.session_state.get("active_dataset_name", "Unnamed Dataset")
target = model_state.get("target", st.session_state.get("target_column", "N/A"))
features = model_state.get("features", st.session_state.get("feature_columns", []))
task_type = model_state.get("task_type", "unknown")
best_model_name = model_state.get("best_model_name", "N/A")
metrics = model_state.get("metrics", {})

st.title("📄 Download Project Report")

st.markdown(
    """
This page generates a **structured PDF report** summarizing:

- Dataset overview  
- Model training summary  
- Feature engineering insights  
- Key recommendations  
"""
)

# ------------- Helper: build PDF in memory -------------
def build_pdf_report() -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # ---- Title ----
    title = f"InsightCure Analytics Report – {dataset_name}"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    # ---- Meta ----
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph(f"Generated on: {now_str}", styles["Normal"]))
    story.append(Paragraph(f"Task Type: {task_type.title()}", styles["Normal"]))
    story.append(Paragraph(f"Best Model: {best_model_name}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # ---- Dataset Overview ----
    story.append(Paragraph("<b>1. Dataset Overview</b>", styles["Heading2"]))
    n_rows, n_cols = df.shape
    story.append(Paragraph(f"Dataset: {dataset_name}", styles["Normal"]))
    story.append(Paragraph(f"Rows: {n_rows}", styles["Normal"]))
    story.append(Paragraph(f"Columns: {n_cols}", styles["Normal"]))

    # Column types
    col_info = df.dtypes.reset_index()
    col_info.columns = ["Column", "Dtype"]
    # Only show first 12 for brevity
    col_info = col_info.head(12)

    data = [["Column", "Type"]] + col_info.values.tolist()
    table = Table(data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))

    # ---- Model Summary ----
    story.append(Paragraph("<b>2. Model Summary</b>", styles["Heading2"]))
    story.append(Paragraph(f"Target variable: <b>{target}</b>", styles["Normal"]))
    story.append(
        Paragraph(
            f"Number of features used: <b>{len(features)}</b>", styles["Normal"]
        )
    )

    if features:
        story.append(
            Paragraph(
                "Features: " + ", ".join([str(f) for f in features]), styles["Normal"]
            )
        )

    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Model Metrics</b>", styles["Heading3"]))

    # Flatten metrics dict into table: Model | Metric | Value
    metrics_rows = [["Model", "Metric", "Value"]]
    for model_name, metric_dict in metrics.items():
        for metric_name, value in metric_dict.items():
            metrics_rows.append(
                [model_name, metric_name, f"{value:.4f}" if isinstance(value, (int, float)) else str(value)]
            )

    if len(metrics_rows) > 1:
        metrics_table = Table(metrics_rows, hAlign="LEFT")
        metrics_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ]
            )
        )
        story.append(metrics_table)
    else:
        story.append(Paragraph("No metrics available.", styles["Normal"]))

    story.append(Spacer(1, 12))

    # ---- Feature Engineering Insights ----
    story.append(Paragraph("<b>3. Feature Engineering Insights</b>", styles["Heading2"]))

    try:
        if features and target in df.columns:
            suggestions = generate_feature_engineering_suggestions(df, target, features)

            # High MI
            story.append(Paragraph("<b>High Importance Features:</b>", styles["Heading3"]))
            high_mi = suggestions.get("high_mi_features", [])
            if high_mi:
                for item in high_mi:
                    story.append(
                        Paragraph(
                            f"- {item.get('feature')} (MI={item.get('mi'):.4f})",
                            styles["Normal"],
                        )
                    )
            else:
                story.append(Paragraph("- None detected.", styles["Normal"]))
            story.append(Spacer(1, 4))

            # Low MI
            story.append(Paragraph("<b>Low Importance Features:</b>", styles["Heading3"]))
            low_mi = suggestions.get("low_mi_features", [])
            if low_mi:
                for item in low_mi:
                    story.append(
                        Paragraph(
                            f"- {item.get('feature')} (MI={item.get('mi'):.4f})",
                            styles["Normal"],
                        )
                    )
            else:
                story.append(Paragraph("- None detected.", styles["Normal"]))
            story.append(Spacer(1, 4))

            # Missing values
            story.append(Paragraph("<b>Missing Value Warnings:</b>", styles["Heading3"]))
            mv_list = suggestions.get("missing_value_warnings", [])
            if mv_list:
                for txt in mv_list:
                    story.append(Paragraph(f"- {txt}", styles["Normal"]))
            else:
                story.append(Paragraph("- No missing value issues detected.", styles["Normal"]))
            story.append(Spacer(1, 4))

            # High cardinality
            story.append(
                Paragraph("<b>High Cardinality Columns:</b>", styles["Heading3"])
            )
            hc = suggestions.get("high_cardinality", [])
            if hc:
                for txt in hc:
                    story.append(Paragraph(f"- {txt}", styles["Normal"]))
            else:
                story.append(Paragraph("- No high-cardinality issues detected.", styles["Normal"]))
            story.append(Spacer(1, 4))

            # Zero variance
            story.append(Paragraph("<b>Zero Variance Columns:</b>", styles["Heading3"]))
            zv = suggestions.get("zero_variance", [])
            if zv:
                for txt in zv:
                    story.append(Paragraph(f"- {txt}", styles["Normal"]))
            else:
                story.append(Paragraph("- No zero-variance columns detected.", styles["Normal"]))
            story.append(Spacer(1, 4))

            # General recommendations
            story.append(
                Paragraph("<b>General Recommendations:</b>", styles["Heading3"])
            )
            recs = suggestions.get("general_recommendations", [])
            if recs:
                for txt in recs:
                    story.append(Paragraph(f"- {txt}", styles["Normal"]))
            story.append(Spacer(1, 12))

        else:
            story.append(
                Paragraph(
                    "Feature engineering insights could not be generated (missing features or target).",
                    styles["Normal"],
                )
            )
            story.append(Spacer(1, 12))

    except Exception as e:
        story.append(
            Paragraph(
                f"Feature engineering analysis failed: {e}", styles["Normal"]
            )
        )
        story.append(Spacer(1, 12))

    # ---- Key Business Takeaways ----
    story.append(Paragraph("<b>4. Key Takeaways</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            "This model training run demonstrates how InsightCure can quickly "
            "ingest a structured dataset, detect the appropriate task type, benchmark "
            "multiple models, and surface the best-performing approach for the selected target.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "The feature engineering analysis highlights which variables drive the target most, "
            "where data quality improvements are needed, and which columns may be removed or transformed "
            "to simplify the model while maintaining performance.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "These insights can guide follow-up experimentation, support stakeholder communication, "
            "and serve as a starting point for productionizing the ML pipeline.",
            styles["Normal"],
        )
    )

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


# ----------------- UI: Generate & Download -----------------
if st.button("Generate PDF Report"):
    with st.spinner("Building PDF report..."):
        pdf_bytes = build_pdf_report()

    st.success("Report generated successfully!")

    file_name = f"InsightCure_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

    st.download_button(
        label="Download Report (PDF)",
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
    )
