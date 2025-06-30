import streamlit as st

st.set_page_config(page_title="Contacts", page_icon="📄", layout="wide")

st.title("Contacts")

st.sidebar.file_uploader(
    "Upload file (xlsx or csv)", type=["xlsx", "csv"], key="page2"
)

