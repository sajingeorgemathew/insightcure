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

# ✅ NEW: CDN VIDEO URL FOR LOGIN BACKGROUND
# Replace this with your *direct* video URL (must end with .mp4 or similar)
VIDEO_BG_URL = "<blockquote class="imgur-embed-pub" lang="en" data-id="a/DeXs9Uy" data-context="false" ><a href="//imgur.com/a/DeXs9Uy"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>"  # <-- put the real direct video link here


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
# 5. Custom Page Loader (InsightCure…)
# ---------------------------------------------------

# ---------------------------------------------------
# 6. LOGIN PAGE (Video + Music)
# ---------------------------------------------------
if not st.session_state.admin_logged_in:

    # ✅ UPDATED: use VIDEO_BG_URL instead of /static/admin_bg.mp4
    st.markdown(
        f"""
        <video autoplay muted loop playsinline id="adminVideoBG">
            <source src="{VIDEO_BG_URL}" type="video/mp4">
        </video>

        <style>
        #adminVideoBG {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            object-fit: cover;
            z-index: -2;
        }}
        .overlay {{
            position: fixed;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            background: rgba(0,0,0,0.55);
            z-index: -1;
        }}
        </style>

        <div class="overlay"></div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <audio autoplay loop>
            <source src="/static/theme_music.mp3" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='text-align:center; color:white;'>Admin Login</h1>",
                unsafe_allow_html=True)

    password = st.text_input("Enter Password", type="password")

    if st.button("Login"):
        if hashlib.sha256(password.encode()).hexdigest() == ADMIN_PASSWORD_HASH:
            st.session_state.admin_logged_in = True
            st.rerun()
        else:
            st.error("Incorrect password")

    st.stop()

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

st.sidebar.markdown("### Navigation (top-left page selector)")
st.sidebar.markdown("""
- Upload Data
- Model Training
- Visualizations
- AI Insights
- Datasets Overview
""")

# Dashboard
st.title("Enterprise Analytics Dashboard")

total_files = len(st.session_state.datasets)
total_rows = sum(df.shape[0] for df in st.session_state.datasets.values())
total_columns = len(
    set().union(*[set(df.columns) for df in st.session_state.datasets.values()])
) if total_files > 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="card"><div class="card-title">Datasets Loaded</div>'
                f'<div class="card-value">{total_files}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card"><div class="card-title">Total Rows</div>'
                f'<div class="card-value">{total_rows}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="card"><div class="card-title">Unique Columns</div>'
                f'<div class="card-value">{total_columns}</div></div>', unsafe_allow_html=True)

st.markdown("---")

if total_files == 0:
    st.info("No datasets loaded yet. Go to **Upload Data** page.")
else:
    st.subheader("📁 Loaded Datasets")
    for name, df in st.session_state.datasets.items():
        st.markdown(f"**{name}** — {df.shape[0]} rows, {df.shape[1]} columns")
