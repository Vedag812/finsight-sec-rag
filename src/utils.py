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
    value = os.getenv(key)
    if value:
        return value

    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass

    return default
