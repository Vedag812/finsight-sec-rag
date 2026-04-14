"""
Utility helpers shared across the app.

get_secret() reads from os.getenv first (local .env file),
then falls back to st.secrets (Streamlit Cloud deployment).
"""

import os


def get_secret(key: str, default: str = "") -> str:
    """
    Read a secret from environment variables OR Streamlit Cloud secrets.

    Priority:
      1. os.getenv (works locally with .env via python-dotenv)
      2. st.secrets (works on Streamlit Community Cloud)
      3. default value
    """
    # 1. Try environment variable (local dev with .env)
    value = os.getenv(key)
    if value:
        return value

    # 2. Try Streamlit secrets (cloud deployment)
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return default
