import streamlit as st

st.set_page_config(page_title="Page 4", page_icon="📄")

st.title("Dataset 4")

st.file_uploader("Upload file (xlsx or csv)", type=["xlsx", "csv"], key="page4")

