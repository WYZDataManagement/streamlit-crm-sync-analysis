import streamlit as st

st.set_page_config(page_title="Orders Fleet", page_icon="WYZ-Etoile-Bleu 1.png", layout="wide")

st.title("Orders Fleet")

st.sidebar.file_uploader(
    "Upload file (xlsx or csv)", type=["xlsx", "csv"], key="page4"
)

