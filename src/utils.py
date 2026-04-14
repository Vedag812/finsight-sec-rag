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
        print(f"  [secret] {key} loaded from env ({value[:4]}...{value[-4:]})")
        return value

    # 2. Try Streamlit secrets (cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            value = str(st.secrets[key]).strip()
            print(f"  [secret] {key} loaded from st.secrets ({value[:4]}...{value[-4:]})")
            return value
    except Exception as e:
        print(f"  [secret] {key} st.secrets failed: {e}")

    print(f"  [secret] {key} NOT FOUND anywhere!")
    return default
