import streamlit as st
import pandas as pd
from pathlib import Path
from modules.theme import load_theme
import hashlib

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
# 5. LOGIN PAGE (InsightCure Built-In Animation)
# ---------------------------------------------------
if not st.session_state.admin_logged_in:

    st.markdown(
        """
        <style>

        /* Background */
        .login-bg {
            background: linear-gradient(135deg, #000000 0%, #1a0000 40%, #300000 100%);
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            z-index: -5;
        }

        /* Center animation box */
        .ic-box {
            width: 70%;
            max-width: 700px;
            margin: 60px auto 20px auto;
            padding: 40px;
            border-radius: 18px;
            text-align: center;
            border: 2px solid rgba(255,0,60,0.6);
            background: rgba(20,0,0,0.45);
            box-shadow: 0 0 25px rgba(255,0,60,0.45);
            animation: glowBox 3s ease-in-out infinite alternate;
        }

        /* Glow animation */
        @keyframes glowBox {
            0% { box-shadow: 0 0 10px rgba(255,0,60,0.3); }
            100% { box-shadow: 0 0 25px rgba(255,0,60,0.8); }
        }

        /* Animated InsightCure text */
        .ic-title {
            font-size: 48px;
            font-weight: 900;
            letter-spacing: 4px;
            color: #ff1a40;
            text-shadow: 0 0 12px rgba(255,0,80,0.9);
            animation: pulseText 2.8s infinite ease-in-out;
        }

        @keyframes pulseText {
            0%   { opacity: 0.4; transform: scale(0.95); }
            50%  { opacity: 1;   transform: scale(1.04); }
            100% { opacity: 0.4; transform: scale(0.95); }
        }

        /* Moving scan bar */
        .scan-bar {
            width: 100%;
            height: 4px;
            margin-top: 18px;
            background: linear-gradient(90deg, transparent, #ff002f, transparent);
            animation: scanMove 1.8s infinite linear;
        }

        @keyframes scanMove {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        </style>

        <div class="login-bg"></div>

        <div class="ic-box">
            <div class="ic-title">InsightCure</div>
            <div class="scan-bar"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------- LOGIN FORM ----------
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

    st.stop()


# ---------------------------------------------------
# 6. MAIN APP (AFTER LOGIN)
# ---------------------------------------------------

# Remove Streamlit menu/footer/header
st.markdown(
    """
    <style>
    div[data-testid="stNotification"] {display:none !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    </style>
    """,
    unsafe_allow_html=True
)

# Inject custom stylesheet AFTER login
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

# KPI Cards
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

# Divider
st.markdown("---")

# Dataset list
if total_files == 0:
    st.info("No datasets loaded yet. Go to **Upload Data** page.")
else:
    st.subheader("📁 Loaded Datasets")
    for name, df in st.session_state.datasets.items():
        st.markdown(f"**{name}** — {df.shape[0]} rows, {df.shape[1]} columns")
