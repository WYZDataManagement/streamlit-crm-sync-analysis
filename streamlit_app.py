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

    st.write("## Filter Data")
    filter_column = st.selectbox("Select column to filter", options=df.columns)
    filter_values = st.multiselect(
        "Select values", options=sorted(df[filter_column].dropna().unique())
    )
    filtered_df = (
        df[df[filter_column].isin(filter_values)] if filter_values else df
    )

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
    fig.update_yaxes(range=[0, null_counts["null_count"].max() + 1])
    st.plotly_chart(fig, use_container_width=True)

    equality_columns = [c for c in df.columns if "IsEquals" in c]
    if equality_columns:
        st.write("## Distribution")
        selected_eq = st.selectbox(
            "Select column for distribution", options=equality_columns
        )
        counts = (
            filtered_df[selected_eq]
            .value_counts(dropna=False)
            .reset_index()
        )
        counts.columns = [selected_eq, "count"]
        fig = px.bar(
            counts,
            x=selected_eq,
            y="count",
            title=f"{selected_eq} distribution",
            color_discrete_sequence=[ASSOCIATED_COLORS[0]],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("## Data Preview")
    st.dataframe(filtered_df)
else:
    st.info("Please upload an Accounts file to see analysis.")

