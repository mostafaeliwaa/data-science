"""Unified Streamlit entry point.

Sidebar nav switches between three modules:
  - Analysis    : marketing-campaign EDA dashboard
  - Chatbot     : Gemini-powered NL-over-DataFrame Q&A
  - Recommendation : pre/in-flight/post-campaign ML models
"""
import os
import sys
import streamlit as st
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv(os.path.join(ROOT, ".env"))

st.set_page_config(
    page_title="Marketing Intelligence Suite",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "📊 Analysis Dashboard": ("analysis", "Marketing campaign EDA & reporting"),
    "💬 Chatbot": ("chatbot", "NL Q&A over your campaign CSV"),
    "🎯 Recommendation Models": ("recommendation", "Pre / In-flight / Post-campaign ML"),
}

st.sidebar.title("📈 Marketing Suite")
st.sidebar.caption("Unified analysis, chatbot, and recommendation app.")

choice = st.sidebar.radio("Module", list(PAGES.keys()), index=0)
module_name, blurb = PAGES[choice]
st.sidebar.info(blurb)
st.sidebar.markdown("---")

if module_name == "analysis":
    from analysis import render
elif module_name == "chatbot":
    from chatbot import render
else:
    from recommendation import render

render()
