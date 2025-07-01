import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv
import numpy as np
from difflib import SequenceMatcher
import re

def load_accounts_file(uploaded_file):
    """Load uploaded Accounts file handling malformed CSV lines."""
    if uploaded_file.name.endswith(".csv"):
        sample = uploaded_file.read(1024)
        uploaded_file.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(sample.decode("utf-8", errors="ignore")).delimiter
        except Exception:
            delimiter = ","

        try:
            df = pd.read_csv(uploaded_file, sep=delimiter, engine="python", on_bad_lines="skip")
        finally:
            uploaded_file.seek(0)
    else:
        df = pd.read_excel(uploaded_file)
    return df

def normalize_bool(val):
    """Return True if val looks like a positive boolean."""
    if pd.isna(val):
        return False
    val_str = str(val).strip().lower()
    return val_str in {"true", "1", "yes", "y", "t", "match"}

def calculate_similarity(text1, text2):
    """Calculate similarity between two texts using SequenceMatcher."""
    if pd.isna(text1) or pd.isna(text2):
        return 0
    return SequenceMatcher(None, str(text1), str(text2)).ratio()

def analyze_text_patterns(series, name):
    """Analyze text patterns in a series."""
    if series.empty:
        return {}
    
    patterns = {
        f'{name}_total_values': len(series),
        f'{name}_null_values': series.isnull().sum(),
        f'{name}_unique_values': series.nunique(),
        f'{name}_empty_strings': (series == '').sum(),
        f'{name}_contains_parentheses': series.str.contains(r'\(.*\)', na=False).sum(),
        f'{name}_all_uppercase': series.str.isupper().sum(),
        f'{name}_contains_numbers': series.str.contains(r'\d', na=False).sum(),
        f'{name}_avg_length': series.str.len().mean() if series.dtype == 'object' else 0,
    }
    return patterns

@st.cache_data
def compute_comprehensive_analysis(data):
    """Compute comprehensive analysis of data quality."""
    eq_cols = [c for c in data.columns if "IsEquals" in c or "IsEqual" in c]
    
    if not eq_cols:
        return {}, {}, pd.DataFrame(), pd.DataFrame(), {}
    
    # Basic statistics
    total_rows = len(data)
    field_stats = []
    similarity_stats = []
    diagnostic_info = {}
    
    # Analyze each comparison field
    for col in eq_cols:
        base_field = col.replace("_IsEquals", "").replace("_IsEqual", "")
        crm_col = f"{base_field}_CRM"
        bi_col = f"{base_field}_BI"
        
        # Get unique values in IsEquals column for debugging
        unique_values = data[col].value_counts().to_dict()
        diagnostic_info[col] = unique_values
        
        # Calculate matches using improved normalize_bool
        match_series = data[col].apply(normalize_bool)
        match_count = int(match_series.sum())
        
        # Calculate Both Null
        both_null_count = 0
        if crm_col in data.columns and bi_col in data.columns:
            both_null_count = int((data[crm_col].isnull() & data[bi_col].isnull()).sum())
        
        no_match_count = total_rows - match_count - both_null_count
        match_rate = (match_count / total_rows * 100) if total_rows > 0 else 0
        
        field_stats.append({
            'field': base_field,
            'match': match_count,
            'no_match': no_match_count,
            'both_null': both_null_count,
            'match_rate': round(match_rate, 2)
        })
        
        # Calculate similarity for text fields (sample of 1000 for performance)
        if crm_col in data.columns and bi_col in data.columns:
            sample_size = min(1000, len(data))
            sample_data = data.sample(n=sample_size) if len(data) > sample_size else data
            
            similarities = sample_data.apply(
                lambda row: calculate_similarity(row[crm_col], row[bi_col]), axis=1
            )
            
            similarity_stats.append({
                'field': base_field,
                'avg_similarity': round(similarities.mean(), 3),
                'min_similarity': round(similarities.min(), 3),
                'max_similarity': round(similarities.max(), 3),
                'similarity_std': round(similarities.std(), 3)
            })
    
    field_stats_df = pd.DataFrame(field_stats)
    similarity_stats_df = pd.DataFrame(similarity_stats)
    
    # Calculate overall stats
    overall_stats = {
        'total_match': field_stats_df['match'].sum() if not field_stats_df.empty else 0,
        'total_no_match': field_stats_df['no_match'].sum() if not field_stats_df.empty else 0,
        'total_both_null': field_stats_df['both_null'].sum() if not field_stats_df.empty else 0,
    }
    
    # Row-level analysis
    if eq_cols:
        bool_df = data[eq_cols].applymap(normalize_bool)
        no_match_per_row = (~bool_df).sum(axis=1)
        rows_with_nomatch = int((no_match_per_row > 0).sum())
        nomatch_dist = no_match_per_row.value_counts().sort_index().reset_index()
        nomatch_dist.columns = ["no_match_count", "row_count"]
    else:
        nomatch_dist = pd.DataFrame()
        rows_with_nomatch = 0
    
    return overall_stats, {'rows_with_nomatch': rows_with_nomatch}, field_stats_df, similarity_stats_df, diagnostic_info, nomatch_dist

def create_diagnostic_dashboard(df, diagnostic_info):
    """Create diagnostic dashboard for debugging data quality issues."""
    st.markdown("## 🚨 Diagnostic d'urgence")
    
    # Show raw values in IsEquals columns
    st.markdown("### Valeurs brutes dans les colonnes de comparaison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top 10 des valeurs uniques par colonne:**")
        for col, values in list(diagnostic_info.items())[:5]:
            with st.expander(f"📊 {col}"):
                st.json(dict(list(values.items())[:10]))
    
    with col2:
        # Sample data comparison
        st.markdown("**Échantillon de données CRM vs BI:**")
        eq_cols = [c for c in df.columns if "IsEquals" in c]
        if eq_cols:
            sample_cols = []
            for col in eq_cols[:3]:
                base = col.replace("_IsEquals", "")
                crm_col = f"{base}_CRM"
                bi_col = f"{base}_BI"
                if crm_col in df.columns and bi_col in df.columns:
                    sample_cols.extend([crm_col, bi_col, col])
            
            if sample_cols:
                st.dataframe(df[sample_cols].head(10))

def create_quality_metrics_dashboard(field_stats_df, overall_stats):
    """Create quality metrics dashboard."""
    st.markdown("## 📊 Métriques de qualité globales")
    
    # KPI Cards
    total_comparisons = sum(overall_stats.values())
    match_rate = (overall_stats['total_match'] / total_comparisons * 100) if total_comparisons > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Taux de correspondance global", f"{match_rate:.1f}%")
    with col2:
        st.metric("✅ Total correspondances", f"{overall_stats['total_match']:,}")
    with col3:
        st.metric("❌ Total écarts", f"{overall_stats['total_no_match']:,}")
    with col4:
        st.metric("⚪ Valeurs nulles", f"{overall_stats['total_both_null']:,}")
    
    # Overall distribution pie chart
    pie_data = pd.DataFrame([{
        'status': 'Correspondances',
        'count': overall_stats['total_match'],
        'color': '#2E8B57'
    }, {
        'status': 'Écarts',
        'count': overall_stats['total_no_match'],
        'color': '#DC143C'
    }, {
        'status': 'Valeurs nulles',
        'count': overall_stats['total_both_null'],
        'color': '#808080'
    }])
    
    fig_pie = px.pie(pie_data, names='status', values='count',
                     title="🥧 Répartition globale de la qualité des données",
                     color='status',
                     color_discrete_map={
                         'Correspondances': '#2E8B57',
                         'Écarts': '#DC143C',
                         'Valeurs nulles': '#808080'
                     })
    fig_pie.update_traces(hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

def create_field_analysis_charts(field_stats_df):
    """Create detailed field analysis charts."""
    st.markdown("## 📈 Analyse détaillée par champ")
    
    if field_stats_df.empty:
        st.warning("Aucune donnée d'analyse de champ disponible")
        return
    
    # Match rate by field
    fig_match_rate = px.bar(
        field_stats_df.sort_values('match_rate'),
        x='match_rate',
        y='field',
        orientation='h',
        title="📊 Taux de correspondance par champ (%)",
        color='match_rate',
        color_continuous_scale=['#DC143C', '#FFD700', '#2E8B57'],
        text='match_rate'
    )
    fig_match_rate.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_match_rate.update_layout(height=max(400, len(field_stats_df) * 30))
    st.plotly_chart(fig_match_rate, use_container_width=True)
    
    # Stacked bar chart - detailed status
    status_data = []
    for _, row in field_stats_df.iterrows():
        status_data.extend([
            {'field': row['field'], 'status': 'Correspondances', 'count': row['match']},
            {'field': row['field'], 'status': 'Écarts', 'count': row['no_match']},
            {'field': row['field'], 'status': 'Valeurs nulles', 'count': row['both_null']}
        ])
    
    status_df = pd.DataFrame(status_data)
    
    fig_stacked = px.bar(
        status_df,
        x='field',
        y='count',
        color='status',
        title="📚 Répartition détaillée par champ",
        color_discrete_map={
            'Correspondances': '#2E8B57',
            'Écarts': '#DC143C',
            'Valeurs nulles': '#808080'
        }
    )
    fig_stacked.update_xaxes(tickangle=45)
    st.plotly_chart(fig_stacked, use_container_width=True)

def create_similarity_analysis(similarity_stats_df):
    """Create similarity analysis charts."""
    if similarity_stats_df.empty:
        return
        
    st.markdown("## 🔍 Analyse de similarité textuelle")
    
    # Similarity scores by field
    fig_sim = px.bar(
        similarity_stats_df.sort_values('avg_similarity'),
        x='avg_similarity',
        y='field',
        orientation='h',
        title="📝 Score de similarité moyen par champ",
        color='avg_similarity',
        color_continuous_scale='RdYlGn',
        text='avg_similarity'
    )
    fig_sim.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_sim.update_layout(height=max(400, len(similarity_stats_df) * 30))
    st.plotly_chart(fig_sim, use_container_width=True)
    
    # Similarity distribution
    fig_sim_dist = px.scatter(
        similarity_stats_df,
        x='avg_similarity',
        y='similarity_std',
        size='max_similarity',
        hover_data=['field'],
        title="🎯 Distribution de la similarité (Moyenne vs Écart-type)",
        labels={
            'avg_similarity': 'Similarité moyenne',
            'similarity_std': 'Écart-type de similarité'
        }
    )
    st.plotly_chart(fig_sim_dist, use_container_width=True)

def create_advanced_analytics(df, field_stats_df):
    """Create advanced analytics and recommendations."""
    st.markdown("## 🎯 Analyses avancées et recommandations")
    
    if field_stats_df.empty:
        return
    
    # Priority matrix - Impact vs Ease of fix
    field_stats_df = field_stats_df.copy()
    field_stats_df['total_records'] = field_stats_df['match'] + field_stats_df['no_match'] + field_stats_df['both_null']
    field_stats_df['error_rate'] = (field_stats_df['no_match'] / field_stats_df['total_records'] * 100).fillna(0)
    
    # Simulate complexity scores (in real app, this would be based on field analysis)
    complexity_map = {
        'name_company': 2, 'address1_city': 2, 'address1_country': 1,
        'emailaddress1': 3, 'telephone1': 3, 'statecode': 1
    }
    
    field_stats_df['fix_complexity'] = field_stats_df['field'].map(
        lambda x: complexity_map.get(x, np.random.randint(1, 5))
    )
    field_stats_df['business_impact'] = field_stats_df['error_rate'] / 20  # Normalize to 0-5 scale
    
    # Priority matrix
    fig_priority = px.scatter(
        field_stats_df,
        x='fix_complexity',
        y='business_impact',
        size='no_match',
        hover_data=['field', 'error_rate'],
        title="🎯 Matrice de priorisation (Impact métier vs Complexité de correction)",
        labels={
            'fix_complexity': 'Complexité de correction (1=facile, 5=difficile)',
            'business_impact': 'Impact métier'
        }
    )
    
    # Add quadrant lines
    fig_priority.add_hline(y=field_stats_df['business_impact'].median(), 
                          line_dash="dash", line_color="gray")
    fig_priority.add_vline(x=field_stats_df['fix_complexity'].median(), 
                          line_dash="dash", line_color="gray")
    
    # Add quadrant labels
    fig_priority.add_annotation(x=1.2, y=4.5, text="🚨 URGENT<br>(Facile + Impact élevé)", 
                               bgcolor="rgba(255,0,0,0.1)")
    fig_priority.add_annotation(x=4.8, y=4.5, text="🎯 PLANIFIER<br>(Difficile + Impact élevé)", 
                               bgcolor="rgba(255,165,0,0.1)")
    fig_priority.add_annotation(x=1.2, y=0.5, text="✅ QUICK WINS<br>(Facile + Impact faible)", 
                               bgcolor="rgba(0,255,0,0.1)")
    fig_priority.add_annotation(x=4.8, y=0.5, text="⏳ REPORTER<br>(Difficile + Impact faible)", 
                               bgcolor="rgba(128,128,128,0.1)")
    
    st.plotly_chart(fig_priority, use_container_width=True)
    
    # Recommendations table
    st.markdown("### 💡 Recommandations de correction")
    
    # Sort by priority (high impact, low complexity first)
    field_stats_df['priority_score'] = field_stats_df['business_impact'] / (field_stats_df['fix_complexity'] + 0.1)
    recommendations = field_stats_df.sort_values('priority_score', ascending=False)
    
    for idx, row in recommendations.head(5).iterrows():
        with st.expander(f"🔧 {row['field']} - Taux d'erreur: {row['error_rate']:.1f}%"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Enregistrements en erreur", f"{row['no_match']:,}")
            with col2:
                st.metric("Complexité (1-5)", f"{row['fix_complexity']}")
            with col3:
                st.metric("Score de priorité", f"{row['priority_score']:.2f}")
            
            # Specific recommendations based on field type
            if 'address' in row['field'].lower():
                st.info("💡 **Suggestion**: Normalisation automatique des adresses (suppression espaces, majuscules)")
            elif 'email' in row['field'].lower():
                st.info("💡 **Suggestion**: Validation format email + normalisation casse")
            elif 'phone' in row['field'].lower() or 'telephone' in row['field'].lower():
                st.info("💡 **Suggestion**: Normalisation format téléphone (suppression espaces, tirets)")
            else:
                st.info("💡 **Suggestion**: Analyse manuelle d'échantillon pour définir règles de nettoyage")

def create_heatmap_visualization(df):
    """Create heatmap visualization of data quality."""
    eq_cols = [c for c in df.columns if "IsEquals" in c]
    if not eq_cols:
        return
        
    st.markdown("## 🔥 Heatmap de qualité des données")
    
    # Limit to reasonable number of rows for performance
    max_rows = 1000
    df_sample = df.head(max_rows) if len(df) > max_rows else df
    
    # Create status matrix
    status_matrix = pd.DataFrame(index=df_sample.index)
    
    for col in eq_cols:
        base_field = col.replace("_IsEquals", "")
        match_series = df_sample[col].apply(normalize_bool)
        
        crm_col = f"{base_field}_CRM"
        bi_col = f"{base_field}_BI"
        
        if crm_col in df.columns and bi_col in df.columns:
            both_null = df_sample[crm_col].isnull() & df_sample[bi_col].isnull()
        else:
            both_null = pd.Series(False, index=df_sample.index)
        
        # Create status: 1 = Match, -1 = No Match, 0 = Both Null
        status = pd.Series(-1, index=df_sample.index)  # Default No Match
        status[match_series] = 1  # Match
        status[both_null] = 0  # Both Null
        
        status_matrix[base_field] = status
    
    if not status_matrix.empty:
        fig_heatmap = px.imshow(
            status_matrix.T,  # Transpose to have fields as rows
            color_continuous_scale=[(0, '#DC143C'), (0.5, '#808080'), (1, '#2E8B57')],
            aspect='auto',
            title=f"🔥 Matrice de qualité (échantillon de {len(df_sample)} lignes)",
            labels={'x': 'Index des enregistrements', 'y': 'Champs'},
            zmin=-1, zmax=1
        )
        fig_heatmap.update_layout(height=max(400, len(status_matrix.columns) * 30))
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("""
        **Légende de la heatmap:**
        - 🟢 **Vert**: Correspondance parfaite
        - 🔴 **Rouge**: Pas de correspondance  
        - ⚪ **Gris**: Valeurs nulles dans les deux systèmes
        """)

# Streamlit App Configuration
st.set_page_config(
    page_title="Analyse Qualité des Données CRM vs BI", 
    page_icon="WYZ-Etoile-Bleu 1.png", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color scheme
COLORS = {
    'primary': "#7fbfdc",
    'danger': '#6ba6b6', 
    'warning': '#4cadb4',
    'neutral': '#45b49d',
    'info': '#82b86a'
}


# App Header
st.markdown("""
# 🎯 Analyse de Qualité des Données CRM vs BI
### Diagnostic complet et recommandations pour l'amélioration de la correspondance des données
""")

# Sidebar
with st.sidebar:
    st.markdown("## 📁 Upload de fichier")
    uploaded_file = st.file_uploader(
        "Choisir un fichier de comparaison",
        type=["xlsx", "csv"],
        help="Fichier Excel ou CSV contenant les comparaisons CRM vs BI"
    )
    
    if uploaded_file:
        st.success(f"✅ Fichier chargé: {uploaded_file.name}")
        
        st.markdown("## ⚙️ Options d'analyse")
        show_diagnostic = st.checkbox("🚨 Mode diagnostic", value=True, 
                                    help="Afficher les diagnostics de débogage")
        show_similarity = st.checkbox("🔍 Analyse de similarité", value=True,
                                    help="Calculer la similarité textuelle (peut être lent)")
        show_heatmap = st.checkbox("🔥 Heatmap qualité", value=False,
                                 help="Afficher la heatmap de qualité des données")
        
        max_similarity_samples = st.slider("Échantillon pour similarité", 100, 5000, 1000,
                                         help="Nombre d'enregistrements pour l'analyse de similarité")

# Main App Logic
if uploaded_file is not None:
    try:
        # Load data
        with st.spinner("🔄 Chargement et analyse des données..."):
            df = load_accounts_file(uploaded_file)
            
            # Comprehensive analysis
            overall_stats, row_stats, field_stats_df, similarity_stats_df, diagnostic_info, nomatch_dist = compute_comprehensive_analysis(df)
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total lignes", f"{len(df):,}")
        with col2:
            eq_cols = [c for c in df.columns if "IsEquals" in c]
            st.metric("🔗 Colonnes de comparaison", len(eq_cols))
        with col3:
            st.metric("🚨 Lignes avec écarts", f"{row_stats.get('rows_with_nomatch', 0):,}")
        
        # Diagnostic Dashboard (if enabled)
        if show_diagnostic:
            create_diagnostic_dashboard(df, diagnostic_info)
        
        # Quality Metrics Dashboard
        if overall_stats:
            create_quality_metrics_dashboard(field_stats_df, overall_stats)
        
        # Field Analysis Charts
        if not field_stats_df.empty:
            create_field_analysis_charts(field_stats_df)
        
        # Similarity Analysis (if enabled)
        if show_similarity and not similarity_stats_df.empty:
            create_similarity_analysis(similarity_stats_df)
        
        # Advanced Analytics
        if not field_stats_df.empty:
            create_advanced_analytics(df, field_stats_df)
        
        # Heatmap (if enabled)
        if show_heatmap:
            create_heatmap_visualization(df)
        
        # Row-level distribution
        if not nomatch_dist.empty:
            st.markdown("## 📈 Distribution des écarts par ligne")
            fig_nomatch = px.bar(
                nomatch_dist,
                x="no_match_count",
                y="row_count",
                title="Nombre de lignes par nombre d'écarts",
                color='row_count',
                color_continuous_scale='Blues'
            )
            fig_nomatch.update_xaxes(title="Nombre d'écarts par ligne")
            fig_nomatch.update_yaxes(title="Nombre de lignes")
            st.plotly_chart(fig_nomatch, use_container_width=True)
        
        # Data Preview
        st.markdown("## 👀 Aperçu des données")
        
        # Show summary stats
        with st.expander("📊 Statistiques résumées"):
            st.dataframe(field_stats_df)
        
        # Show raw data sample
        with st.expander("🔍 Échantillon des données brutes"):
            st.dataframe(df.head(20))
        
        # Export results
        st.markdown("## 💾 Export des résultats")
        col1, col2 = st.columns(2)
        
        with col1:
            if not field_stats_df.empty:
                csv_stats = field_stats_df.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger statistiques par champ (CSV)",
                    data=csv_stats,
                    file_name="field_statistics.csv",
                    mime="text/csv"
                )
        
        with col2:
            if not similarity_stats_df.empty:
                csv_similarity = similarity_stats_df.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger analyse de similarité (CSV)",
                    data=csv_similarity,
                    file_name="similarity_analysis.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement du fichier: {str(e)}")
        st.info("💡 Vérifiez le format de votre fichier et réessayez.")
        
        # Debug info
        with st.expander("🔍 Informations de débogage"):
            st.code(f"Type d'erreur: {type(e).__name__}")
            st.code(f"Message: {str(e)}")

else:
    # Welcome screen
    st.markdown("""
    ## 👋 Bienvenue dans l'outil d'analyse de qualité des données
    
    Cet outil vous permet d'analyser la qualité de correspondance entre vos données CRM et BI.
    
    ### 🚀 Pour commencer:
    1. **📁 Uploadez votre fichier** dans la barre latérale (Excel ou CSV)
    2. **⚙️ Configurez les options** d'analyse selon vos besoins
    3. **📊 Explorez les résultats** avec les graphiques interactifs
    
    ### 🎯 Fonctionnalités principales:
    - ✅ **Analyse de correspondance** par champ
    - 🔍 **Calcul de similarité** textuelle avancé
    - 🎯 **Matrice de priorisation** des corrections
    - 🔥 **Heatmap de qualité** interactive
    - 💡 **Recommandations automatiques** de correction
    - 📥 **Export des résultats** en CSV
    
    ### 📋 Format de fichier attendu:
    Le fichier doit contenir des colonnes avec les suffixes:
    - `_CRM` pour les données du système CRM
    - `_BI` pour les données du système BI  
    - `_IsEquals` pour les indicateurs de correspondance
    
    **Exemple**: `name_CRM`, `name_BI`, `name_IsEquals`
    """)
    
    # Sample data structure
    st.markdown("### 📝 Exemple de structure de données:")
    sample_data = pd.DataFrame({
        'name_CRM': ['TOUYRE (LABESSERETTE)', 'PARIZE SAMMUEL'],
        'name_BI': ['TOUYRE (LABESSERETTE)', 'PARIZE SAMMUEL'],
        'name_IsEquals': ['Match', 'Match'],
        'address1_city_CRM': ['LABESSERETTE', 'RUYNES EN MARGERIDE'],
        'address1_city_BI': ['LABESSERETTE', 'RUYNES EN MARGERIDE'],
        'address1_city_IsEquals': ['Match', 'Match']
    })
    st.dataframe(sample_data)