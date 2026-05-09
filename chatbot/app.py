import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from .engine import MarketingAnalyst

load_dotenv()


def _get_analyst():
    if "analyst" in st.session_state:
        return st.session_state.analyst
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Add it to .env and reload.")
        st.stop()
    st.session_state.analyst = MarketingAnalyst(api_key)
    return st.session_state.analyst


def render():
    st.markdown("## 💬 Marketing Campaign Chatbot")
    st.caption(
        "Upload a campaign CSV, then ask questions in English or Arabic. "
        "The model writes pandas code against your dataframe and returns the result."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_df" not in st.session_state:
        st.session_state.chat_df = None

    with st.expander("📤 Upload data", expanded=st.session_state.chat_df is None):
        uploaded = st.file_uploader("CSV file", type=["csv"], key="chat_upload")
        if uploaded is not None:
            analyst = _get_analyst()
            df = pd.read_csv(io.BytesIO(uploaded.read()))
            st.session_state.chat_df = analyst._preprocess_data(df)
            st.success(f"Loaded {len(st.session_state.chat_df):,} rows · {len(st.session_state.chat_df.columns)} columns")

    if st.session_state.chat_df is None:
        st.info("Upload a CSV to start chatting.")
        return

    df = st.session_state.chat_df
    with st.expander("Schema preview"):
        st.dataframe(df.head(5), use_container_width=True)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("🧹 Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col_b:
        if st.button("🔄 Reset data", use_container_width=True):
            st.session_state.chat_df = None
            st.session_state.chat_history = []
            st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if isinstance(content, list):
                st.dataframe(pd.DataFrame(content), use_container_width=True)
            else:
                st.markdown(str(content))

    user_query = st.chat_input("Ask about your campaigns…")
    if not user_query:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    analyst = _get_analyst()
    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            history_for_model = [
                {"role": m["role"], "content": str(m["content"])[:800]}
                for m in st.session_state.chat_history[:-1][-10:]
            ]
            result = analyst.analyze(df, user_query, chat_history=history_for_model)

        if isinstance(result, list):
            st.dataframe(pd.DataFrame(result), use_container_width=True)
        else:
            st.markdown(str(result))

    st.session_state.chat_history.append({"role": "assistant", "content": result})
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]
