"""Legacy entrypoint that forwards users to the Home page."""

import streamlit as st

st.set_page_config(
    page_title="SensorMind Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

try:
    st.switch_page("Home.py")
except Exception:
    st.title("SensorMind Dashboard")
    st.info("Open the main app with: streamlit run app/Home.py")
