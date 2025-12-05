import streamlit as st
import pandas as pd
from pathlib import Path
from modules.theme import load_theme
import hashlib
import base64

# ---------------------------------------------------
# 0. Load global theme (safe)
# ---------------------------------------------------
load_theme()

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
# 5. LOGIN PAGE (Image Background + Overlay)
# ---------------------------------------------------
if not st.session_state.admin_logged_in:

    # --- Path to your background image ---
    # Make sure this file exists in your project: static/login_bg.jpg
    img_path = Path("static/login_bg.jpg")
    if img_path.exists():
        # Read image and encode in base64 so it works reliably
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode()

        background_style = f"""
        <style>
        .login-bg {{
            background-image: url("data:image/jpeg;base64,{img_b64}");
            background-size: cover;
            background-position: center;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            z-index: -5;
        }}
        .overlay {{
            position: fixed;
            width: 100%; height: 100%;
            top: 0; left: 0;
            background: rgba(0,0,0,0.55);
            z-index: -4;
        }}
        .login-container {{
            position: relative;
            z-index: 1;
            padding-top: 120px;
        }}
        </style>
        <div class="login-bg"></div>
        <div class="overlay"></div>
        """
        st.markdown(background_style, unsafe_allow_html=True)

    else:
        # If image not found, fallback to plain background
        st.markdown("<style>body {background-color: #000000;}</style>", unsafe_allow_html=True)

    # --- Login UI ---
    st.markdown("<div class='login-container'></div>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center; color:white;'>Admin Login</h1>", unsafe_allow_html=True)

    password = st.text_input("Enter Password", type="password")

    if st.button("Login"):
        entered_hash = hashlib.sha256(password.encode()).hexdigest()
        if entered_hash == ADMIN_PASSWORD_HASH:
            st.session_state.admin_logged_in = True
            st.rerun()
        else:
            st.error("Incorrect password")

    st.stop()

# ---------------------------------------------------
# 6. MAIN APP (AFTER LOGIN)
# ---------------------------------------------------

# Hide Streamlit UI elements
st.markdown("""
<style>
div[data-testid="stNotification"] {display:none !important;}
footer {visibility: hidden !important;}
header {visibility: hidden !important;}
</style>
""", unsafe_allow_html=True)

# Load custom CSS after login
styles_path = Path("static/styles.css")
if styles_path.exists():
    with open(styles_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
        f'<div class="card"><div class="card-title">Datasets Loaded</div>'
        f'<div class="card-value">{total_files}</div></div>',
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f'<div class="card"><div class="card-title">Total Rows</div>'
        f'<div class="card-value">{total_rows}</div></div>',
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f'<div class="card"><div class="card-title">Unique Columns</div>'
        f'<div class="card-value">{total_columns}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("---")

if total_files == 0:
    st.info("No datasets loaded yet. Go to **Upload Data** page.")
else:
    st.subheader("📁 Loaded Datasets")
    for name, df in st.session_state.datasets.items():
        st.markdown(f"**{name}** — {df.shape[0]} rows, {df.shape[1]} columns")
