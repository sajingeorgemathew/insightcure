import streamlit as st
import pandas as pd
from pathlib import Path
from modules.theme import load_theme
import hashlib

# ---------------------------------------------------
# 0. Load theme normally (safe)
# ---------------------------------------------------
load_theme()

# ---------------------------------------------------
# 1. INITIAL CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="InsightCure Portal", layout="wide")

# ---------------------------------------------------
# 2. Password Hash
# ---------------------------------------------------
ADMIN_PASSWORD_HASH = hashlib.sha256("yourpassword123".encode()).hexdigest()

# ---------------------------------------------------
# 2B. DIRECT CDN VIDEO URL (use MP4-only link)
# ---------------------------------------------------
VIDEO_BG_URL = "https://i.imgur.com/8l7bI5e.mp4"   # <-- replace with your direct MP4


# ---------------------------------------------------
# 3. Session Init
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
# 5. LOGIN PAGE (SAFE VIDEO PANEL + MUSIC)
# ---------------------------------------------------
if not st.session_state.admin_logged_in:

    # --------- STABLE VIDEO PANEL (no fullscreen autoplay issues) ---------
    st.markdown(
        f"""
        <style>
        .login-bg {{
            background: radial-gradient(circle at top,
                rgba(140,0,0,0.65),
                rgba(0,0,0,0.95));
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            z-index: -5;
        }}
        .video-box {{
            width: 70%;
            max-width: 700px;
            border-radius: 18px;
            overflow: hidden;
            margin: 40px auto 20px auto;
            box-shadow: 0 0 25px rgba(255,0,60,0.55);
        }}
        </style>

        <div class="login-bg"></div>

        <div class="video-box">
            <video autoplay muted loop playsinline style="width:100%; height:auto;">
                <source src="{VIDEO_BG_URL}" type="video/mp4">
            </video>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------- BACKGROUND MUSIC (unchanged) ---------
    st.markdown(
        """
        <audio autoplay loop>
            <source src="/static/theme_music.mp3" type="audio/mp3">
        </audio>
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
        if hashlib.sha256(password.encode()).hexdigest() == ADMIN_PASSWORD_HASH:
            st.session_state.admin_logged_in = True
            st.rerun()   # SAFE VERSION OF rerun()
        else:
            st.error("Incorrect password")

    st.stop()  # Prevent loading rest of app until login


# ---------------------------------------------------
# 7. MAIN APP (After Login)
# ---------------------------------------------------

# Hide Streamlit warnings, menu, footer
st.markdown("""
<style>
div[data-testid="stNotification"] {display:none !important;}
footer {visibility: hidden !important;}
header {visibility: hidden !important;}
</style>
""", unsafe_allow_html=True)


# Load CSS AFTER login
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
### Navigation  
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
total_columns = len(
    set().union(*[set(df.columns) for df in st.session_state.datasets.values()])
) if total_files > 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f'<div class="card"><div class="card-title">Datasets Loaded</div>'
        f'<div class="card-value">{total_files}</div></div>',
        unsafe_allow_html=True)
with col2:
    st.markdown(
        f'<div class="card"><div class="card-title">Total Rows</div>'
        f'<div class="card-value">{total_rows}</div></div>',
        unsafe_allow_html=True)
with col3:
    st.markdown(
        f'<div class="card"><div class="card-title">Unique Columns</div>'
        f'<div class="card-value">{total_columns}</div></div>',
        unsafe_allow_html=True)

st.markdown("---")

if total_files == 0:
    st.info("No datasets loaded yet. Go to **Upload Data** page.")
else:
    st.subheader("📁 Loaded Datasets")
    for name, df in st.session_state.datasets.items():
        st.markdown(f"**{name}** — {df.shape[0]} rows, {df.shape[1]} columns")
