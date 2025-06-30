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


@st.cache_data
def compute_field_stats(data: pd.DataFrame):
    """Return per-field match statistics and overall counts."""
    eq_cols = [c for c in data.columns if "IsEquals" in c]
    rows = len(data)
    stats = []
    total_match = total_no_match = total_both_null = 0

    for col in eq_cols:
        base = col.replace("_IsEquals", "")
        crm_col = f"{base}_CRM"
        bi_col = f"{base}_BI"

        match_series = data[col].apply(normalize_bool)
        match_count = int(match_series.sum())
        both_null_count = 0
        if crm_col in data.columns and bi_col in data.columns:
            both_null_count = int(
                (data[crm_col].isnull() & data[bi_col].isnull()).sum()
            )
        no_match_count = rows - match_count - both_null_count

        stats.append(
            {
                "field": base,
                "match": match_count,
                "no_match": no_match_count,
                "both_null": both_null_count,
                "match_rate": round(match_count / rows * 100, 2) if rows else 0,
            }
        )

        total_match += match_count
        total_no_match += no_match_count
        total_both_null += both_null_count

    overall = {
        "match": total_match,
        "no_match": total_no_match,
        "both_null": total_both_null,
    }

    return pd.DataFrame(stats), overall


def compute_heatmap_status(data: pd.DataFrame, eq_cols):
    """Return dataframe of match status for heatmap visualisation."""
    status_df = pd.DataFrame(index=data.index)
    for col in eq_cols:
        base = col.replace("_IsEquals", "")
        crm_col = f"{base}_CRM"
        bi_col = f"{base}_BI"

        match_series = data[col].apply(normalize_bool)
        if crm_col in data.columns and bi_col in data.columns:
            both_null = data[crm_col].isnull() & data[bi_col].isnull()
        else:
            both_null = pd.Series(False, index=data.index)

        status = pd.Series("No Match", index=data.index)
        status[match_series] = "Match"
        status[both_null] = "Both Null"

        status_df[base] = status

    return status_df


def emergency_data_diagnosis(df: pd.DataFrame) -> None:
    """Display emergency diagnostic charts."""

    if "name_CRM" in df.columns and "name_BI" in df.columns:
        df["name_CRM_length"] = df["name_CRM"].astype(str).str.len()
        df["name_BI_length"] = df["name_BI"].astype(str).str.len()
        scatter = px.scatter(
            df,
            x="name_CRM_length",
            y="name_BI_length",
            title="Longueur des noms CRM vs BI",
        )
        st.plotly_chart(scatter, use_container_width=True)

        from difflib import SequenceMatcher

        similarity = df.apply(
            lambda row: SequenceMatcher(None, str(row["name_CRM"]), str(row["name_BI"])).ratio(),
            axis=1,
        )
        similarity_fig = px.histogram(
            similarity,
            nbins=50,
            title="Distribution de similarité (SequenceMatcher)",
        )
        st.plotly_chart(similarity_fig, use_container_width=True)

        patterns = {
            "Contient parenthèses": df["name_CRM"].str.contains(r"\(", na=False).sum(),
            "Tout en majuscules": df["name_CRM"].str.isupper().sum(),
            "Contient chiffres": df["name_CRM"].str.contains(r"\d", na=False).sum(),
            "Valeurs nulles": df["name_CRM"].isnull().sum(),
        }
        pattern_df = pd.DataFrame(list(patterns.items()), columns=["Pattern", "Count"])
        pattern_fig = px.bar(pattern_df, x="Pattern", y="Count", title="Patterns détectés dans name_CRM")
        st.plotly_chart(pattern_fig, use_container_width=True)

    type_counts = df.dtypes.astype(str).value_counts().reset_index()
    type_counts.columns = ["Type", "Count"]
    type_fig = px.bar(type_counts, x="Type", y="Count", title="Distribution des types de données")
    st.plotly_chart(type_fig, use_container_width=True)


if uploaded_file is not None:
    df = load_accounts_file(uploaded_file)

    eq_cols, rows_with_nomatch, nomatch_dist, match_counts = compute_analysis(df)
    field_stats, overall_counts = compute_field_stats(df)
    heatmap_status = compute_heatmap_status(df, eq_cols) if eq_cols else pd.DataFrame()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total rows", len(df))
    col2.metric("Rows with ≥1 no match", rows_with_nomatch)
    col3.metric("Equality columns", len(eq_cols))

    emergency_data_diagnosis(df)

    st.write("## Data Preview")
    st.dataframe(df)
else:
    st.info("Please upload an Accounts file to see analysis.")
