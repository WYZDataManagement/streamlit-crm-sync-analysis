import streamlit as st

st.set_page_config(page_title="Orders Retail", page_icon="📄", layout="wide")

st.title("Orders Retail")

st.sidebar.file_uploader(
    "Upload file (xlsx or csv)", type=["xlsx", "csv"], key="page3"
)

