import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Accounts", page_icon="WYZ-Etoile-Bleu 1.png", layout="wide")

ASSOCIATED_COLORS = [
    "#7fbfdc",
    "#6ba6b6",
    "#4cadb4",
    "#78b495",
    "#82b86a",
    "#45b49d",
]

st.title("Accounts Data Analysis")

uploaded_file = st.sidebar.file_uploader(
    "Upload Accounts file (xlsx or csv)",
    type=["xlsx", "csv"],
    help="Upload an Excel or CSV file with Account data",
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    else:
        df = pd.read_excel(uploaded_file)

    st.write("## Data Preview")
    st.dataframe(df.head())

    st.write("## Null values per column")
    null_counts = df.isnull().sum().reset_index()
    null_counts.columns = ["column", "null_count"]
    fig = px.bar(
        null_counts,
        x="column",
        y="null_count",
        title="Null values by column",
        color_discrete_sequence=ASSOCIATED_COLORS,
    )
    st.plotly_chart(fig, use_container_width=True)

    equality_columns = [
        c
        for c in df.columns
        if "IsEquals" in c or c.endswith("_phone")
    ]

    for i, col in enumerate(equality_columns[:14], start=1):
        counts = df[col].value_counts(dropna=False).reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(
            counts,
            x=col,
            y="count",
            title=f"{col} distribution",
            color_discrete_sequence=[ASSOCIATED_COLORS[i % len(ASSOCIATED_COLORS)]],
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload an Accounts file to see analysis.")

