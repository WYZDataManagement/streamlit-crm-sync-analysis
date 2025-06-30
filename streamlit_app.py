import streamlit as st
import pandas as pd
import plotly.express as px
import csv


def load_accounts_file(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
) -> pd.DataFrame:
    """Load uploaded Accounts file handling malformed CSV lines."""
    if uploaded_file.name.endswith(".csv"):
        sample = uploaded_file.read(1024)
        uploaded_file.seek(0)
        try:
            delimiter = (
                csv.Sniffer().sniff(sample.decode("utf-8", errors="ignore")).delimiter
            )
        except Exception:
            delimiter = ","

        try:
            df = pd.read_csv(
                uploaded_file,
                sep=delimiter,
                engine="python",
                on_bad_lines="skip",
            )
        finally:
            uploaded_file.seek(0)
    else:
        df = pd.read_excel(uploaded_file)

    return df


st.set_page_config(
    page_title="Accounts", page_icon="WYZ-Etoile-Bleu 1.png", layout="wide"
)

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


def normalize_bool(val: object) -> bool:
    """Return True if val looks like a positive boolean."""
    return str(val).strip().lower() in {"true", "1", "yes", "y", "t"}


@st.cache_data
def compute_analysis(data: pd.DataFrame):
    eq_cols = [c for c in data.columns if "IsEquals" in c]
    if eq_cols:
        bool_df = data[eq_cols].applymap(normalize_bool)
        no_match_per_row = (~bool_df).sum(axis=1)
        rows_with_nomatch = int((no_match_per_row > 0).sum())
        nomatch_dist = no_match_per_row.value_counts().sort_index().reset_index()
        nomatch_dist.columns = ["no_match_count", "row_count"]
        match_counts = (
            bool_df.apply(lambda s: s.value_counts(dropna=False))
            .T.fillna(0)
            .reset_index()
        )
        match_counts = match_counts.rename(columns={"index": "column"})
    else:
        bool_df = pd.DataFrame()
        no_match_per_row = pd.Series(dtype=int)
        rows_with_nomatch = 0
        nomatch_dist = pd.DataFrame(columns=["no_match_count", "row_count"])
        match_counts = pd.DataFrame(columns=["column", True, False])
    return eq_cols, rows_with_nomatch, nomatch_dist, match_counts


if uploaded_file is not None:
    df = load_accounts_file(uploaded_file)

    eq_cols, rows_with_nomatch, nomatch_dist, match_counts = compute_analysis(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total rows", len(df))
    col2.metric("Rows with ≥1 no match", rows_with_nomatch)
    col3.metric("Equality columns", len(eq_cols))

    if not nomatch_dist.empty:
        st.write("## No match counts per row")
        fig = px.bar(
            nomatch_dist,
            x="no_match_count",
            y="row_count",
            title="Rows by number of no matches",
            color_discrete_sequence=[ASSOCIATED_COLORS[0]],
        )
        fig.update_xaxes(title="Number of no matches")
        fig.update_yaxes(title="Row count")
        st.plotly_chart(fig, use_container_width=True)

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

    if not match_counts.empty:
        st.write("## Match vs no match by equality column")
        match_melt = match_counts.melt(
            id_vars="column", value_name="count", var_name="match"
        )
        fig = px.bar(
            match_melt,
            x="column",
            y="count",
            color="match",
            barmode="group",
            title="Match vs No Match counts",
            color_discrete_sequence=ASSOCIATED_COLORS[:2],
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.sidebar:
        selected_eq = st.selectbox("Select column for distribution", options=eq_cols)
        apply_distribution = st.button("Apply")

    if eq_cols and apply_distribution:
        counts = df[selected_eq].astype(str).value_counts(dropna=False).reset_index()
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
    st.dataframe(df)
else:
    st.info("Please upload an Accounts file to see analysis.")
