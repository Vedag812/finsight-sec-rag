"""
Utility helpers shared across the app.

get_secret() reads from multiple sources to ensure secrets work
both locally (.env) and on Streamlit Community Cloud (st.secrets).
"""

import os


def get_secret(key: str, default: str = "") -> str:
    """
    Read a secret — works on local dev and Streamlit Cloud.

    Checks in order:
      1. os.environ  (set by load_dotenv locally, or by Streamlit Cloud)
      2. st.secrets  (Streamlit Cloud's TOML-based secret store)
      3. default
    """
    # 1. env var (local .env or Streamlit injected)
    value = os.environ.get(key, "")
    if value:
        return value

    # 2. Streamlit secrets (TOML)
    try:
        import streamlit as st
        value = st.secrets.get(key, "")
        if value:
            return str(value).strip()
    except Exception:
        pass

    # 3. If Streamlit injects secrets as env AFTER import,
    #    force-inject them now
    try:
        import streamlit as st
        for k, v in st.secrets.items():
            if isinstance(v, str):
                os.environ[k] = v
        value = os.environ.get(key, "")
        if value:
            return value
    except Exception:
        pass

    return default
