import streamlit as st
import pandas as pd
from pathlib import Path
from modules.theme import load_theme
import hashlib
import base64

# ---------------------------------------------------
# 1. INITIAL CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="InsightCure Portal", layout="wide")

# ---------------------------------------------------
# 2. Admin Password Hash
# ---------------------------------------------------
ADMIN_PASSWORD_HASH = hashlib.sha256("yourpassword123".encode()).hexdigest()

# ---------------------------------------------------
# 3. Session Initialization
# ---------------------------------------------------
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if "datasets" not in st.session_state:
    st.session_state.datasets = {}

# ---------------------------------------------------
# 4. Global Click Sound
# ---------------------------------------------------
st.markdown(
    """
    <script>
    document.addEventListener("click", function() {
        var audio = new Audio("/static/click.mp3");
        audio.volume = 0.45;
        audio.play();
    });
    </script>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# 5. LOGIN PAGE (Background Image + Styled Input)
# ---------------------------------------------------
if not st.session_state.admin_logged_in:

    img_path = Path("static/login_bg.jpg")
    if img_path.exists():
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode()

        st.markdown(
            f"""
            <style>

            /* Fullscreen background image */
            body {{
                background-image: url("data:image/jpeg;base64,{img_b64}");
                background-size: cover !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-attachment: fixed !important;
            }}

            /* Remove white header margin */
            .stApp {{
                background: transparent !important;
                margin-top: 0 !important;
                padding-top: 0 !important;
            }}

            /* Dark overlay for clarity */
            .login-overlay {{
                position: fixed;
                width: 100%; height: 100%;
                top: 0; left: 0;
                background: rgba(0,0,0,0.55);
                z-index: 0;
            }}

            /* Center login section */
            .login-container {{
                z-index: 1;
                position: relative;
                padding-top: 140px;
            }}

            /* Make password box white text */
            input[type="password"] {{
                color: white !important;
                font-weight: 600;
                background-color: rgba(0,0,0,0.35) !important;
                border: 1px solid #ffffff55 !important;
                border-radius: 10px !important;
            }}

            /* Placeholder text white */
            input::placeholder {{
                color: #f0f0f0 !important;
                opacity: 1 !important;
            }}

            /* Sidebar not visible on login */
            section[data-testid="stSidebar"] {{
                display: none;
            }}

            </style>

            <div class="login-overlay"></div>
            """,
            unsafe_allow_html=True
        )

    # Login UI
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center; color:white;'>Admin Login</h1>",
        unsafe_allow_html=True
    )

    password = st.text_input("Enter Password", type="password")

    if st.button("Login"):
        entered_hash = hashlib.sha256(password.encode()).hexdigest()
        if entered_hash == ADMIN_PASSWORD_HASH:
            st.session_state.admin_logged_in = True
            st.rerun()
        else:
            st.error("Incorrect password")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------
# 6. MAIN APP (AFTER LOGIN)
# ---------------------------------------------------

# Load theme ONLY after login
load_theme()

# Hide top menu + footer
st.markdown(
    """
    <style>

    div[data-testid="stNotification"] {display:none !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}

    /* Sidebar custom color */
    section[data-testid="stSidebar"] {
        background-color: #0A0A0A !important;
        border-right: 1px solid #222 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("InsightCure FPP Portal")
st.sidebar.markdown("Your AI-powered analytics workspace.")

if Path("static/logo.png").exists():
    st.sidebar.image("static/logo.png", use_column_width=True)

st.sidebar.markdown(
    """
    ### Navigation (top-left page selector)
    - Upload Data
    - Model Training
    - Visualizations
    - AI Insights
    - Datasets Overview
    """
)

# Dashboard
st.title("Enterprise Analytics Dashboard")

total_files = len(st.session_state.datasets)
total_rows = sum(df.shape[0] for df in st.session_state.datasets.values())
total_columns = (
    len(set().union(*[set(df.columns) for df in st.session_state.datasets.values()]))
    if total_files > 0 else 0
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Datasets Loaded</div>
            <div class="card-value">{total_files}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Total Rows</div>
            <div class="card-value">{total_rows}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Unique Columns</div>
            <div class="card-value">{total_columns}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

if total_files == 0:
    st.info("No datasets loaded yet. Go to **Upload Data** page.")
else:
    st.subheader("📁 Loaded Datasets")
    for name, df in st.session_state.datasets.items():
        st.markdown(f"**{name}** — {df.shape[0]} rows, {df.shape[1]} columns")
