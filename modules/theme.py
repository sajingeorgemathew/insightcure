import streamlit as st
from pathlib import Path

def load_theme():
    """Load theme only AFTER login."""
    # Theme should NOT load before login
    if "admin_logged_in" in st.session_state and st.session_state.admin_logged_in:
        css_path = Path("static/styles.css")
        if css_path.exists():
            with open(css_path) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
