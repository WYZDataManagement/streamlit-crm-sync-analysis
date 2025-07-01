import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv
import numpy as np
from difflib import SequenceMatcher
import re
from functools import lru_cache

# Configuration de la page pour optimiser les performances
st.set_page_config(
    page_title="Analyse Qualité des Données CRM vs BI", 
    page_icon="WYZ-Etoile-Bleu 1.png", 
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(show_spinner=False)
def load_accounts_file(uploaded_file_bytes, file_name):
    """Load uploaded Accounts file handling malformed CSV lines - optimized with caching."""
    # Convertir les bytes en DataFrame
    if file_name.endswith(".csv"):
        try:
            # Essayer UTF-8 d'abord, puis ISO-8859-1 si ça échoue
            try:
                content = uploaded_file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                content = uploaded_file_bytes.decode('iso-8859-1')
            
            # Détecter le délimiteur rapidement
            first_line = content.split('\n')[0]
            delimiter = ',' if ',' in first_line else ';' if ';' in first_line else '\t'
            
            # Lecture optimisée
            from io import StringIO
            df = pd.read_csv(StringIO(content), sep=delimiter, engine="c", low_memory=False)
        except Exception:
            # Fallback vers pandas standard
            from io import BytesIO
            df = pd.read_csv(BytesIO(uploaded_file_bytes), engine="python", on_bad_lines="skip")
    else:
        from io import BytesIO
        df = pd.read_excel(BytesIO(uploaded_file_bytes), engine='openpyxl')
    
    return df

@lru_cache(maxsize=1000)
def normalize_bool_cached(val_str):
    """Cached version of normalize_bool for better performance."""
    if pd.isna(val_str) or val_str == '':
        return False
    return str(val_str).strip().lower() in {"true", "1", "yes", "y", "t", "match"}

def normalize_bool_vectorized(series):
    """Vectorized version of normalize_bool for much better performance."""
    # Convertir en string et nettoyer
    str_series = series.astype(str).str.strip().str.lower()
    # Utiliser isin pour une opération vectorisée - exclure "both null"
    return str_series.isin({"true", "1", "yes", "y", "t", "match"}) & ~str_series.isin({"both null", "bothnull", "null", "both_null"})

@lru_cache(maxsize=500)
def calculate_similarity_cached(text1, text2):
    """Cached similarity calculation."""
    if pd.isna(text1) or pd.isna(text2) or text1 == '' or text2 == '':
        return 0
    return SequenceMatcher(None, str(text1), str(text2)).ratio()

@lru_cache(maxsize=100)
def get_field_display_name(col):
    """Convert IsEquals column name to display format like name(CRM)/company(BI) - cached."""
    base_field = col.replace("_IsEquals", "").replace("_IsEqual", "")
    
    parts = base_field.split('_')
    if len(parts) >= 2:
        first_field = '_'.join(parts[:-1])
        second_field = parts[-1]
        return f"{first_field}(CRM)/{second_field}(BI)"
    else:
        return f"{base_field}(CRM)/{base_field}(BI)"

@st.cache_data(show_spinner=False)
def compute_comprehensive_analysis(data):
    """Compute comprehensive analysis of data quality - heavily optimized."""
    eq_cols = [c for c in data.columns if "IsEquals" in c or "IsEqual" in c]
    
    if not eq_cols:
        return {}, {}, pd.DataFrame(), pd.DataFrame()
    
    total_rows = len(data)
    field_stats = []
    
    # Optimisation : traitement vectorisé pour les lignes avec écarts - lecture directe
    rows_with_nomatch = 0
    if eq_cols:
        # Vectorisation complète en lisant directement les valeurs IsEquals
        has_nomatch_per_row = pd.Series(False, index=data.index)
        
        for col in eq_cols:
            # Lecture directe des valeurs "No Match"
            col_values = data[col].astype(str).str.strip().str.lower()
            no_match_series = col_values.isin({"no match", "nomatch", "no_match", "false", "0", "no", "n", "f"})
            has_nomatch_per_row |= no_match_series
        
        rows_with_nomatch = int(has_nomatch_per_row.sum())
    
    rows_iso = total_rows - rows_with_nomatch
    
    # Traitement optimisé pour chaque champ
    for col in eq_cols:
        base_field = col.replace("_IsEquals", "").replace("_IsEqual", "")
        crm_col = f"{base_field}_CRM"
        bi_col = f"{base_field}_BI"
        
        # Utilisation de la version vectorisée pour les matches
        match_series = normalize_bool_vectorized(data[col])
        match_count = int(match_series.sum())
        
        # Calcul des Both Null : vérifier d'abord la colonne IsEquals
        both_null_series = data[col].astype(str).str.strip().str.lower().isin(['both null', 'bothnull', 'null', 'both_null'])
        both_null_count = int(both_null_series.sum())
        
        # Si pas de "Both Null" explicite, vérifier les colonnes CRM/BI
        if both_null_count == 0 and crm_col in data.columns and bi_col in data.columns:
            both_null_count = int((data[crm_col].isnull() & data[bi_col].isnull()).sum())
        
        no_match_count = total_rows - match_count - both_null_count
        match_rate = (match_count / total_rows * 100) if total_rows > 0 else 0
        
        display_name = get_field_display_name(col)
        
        field_stats.append({
            'field': display_name,
            'match': match_count,
            'no_match': no_match_count,
            'both_null': both_null_count,
            'match_rate': round(match_rate, 2)
        })
    
    field_stats_df = pd.DataFrame(field_stats)
    
    # Calcul global optimisé
    overall_stats = {
        'total_match': int(field_stats_df['match'].sum()) if not field_stats_df.empty else 0,
        'total_no_match': int(field_stats_df['no_match'].sum()) if not field_stats_df.empty else 0,
        'total_both_null': int(field_stats_df['both_null'].sum()) if not field_stats_df.empty else 0,
    }
    
    row_stats = {
        'rows_with_nomatch': rows_with_nomatch,
        'rows_iso': rows_iso
    }
    
    return overall_stats, row_stats, field_stats_df, pd.DataFrame()  # Pas de similarity par défaut

@st.cache_data(show_spinner=False)
def compute_similarity_analysis(data, field_stats_df):
    """Compute similarity analysis separately for performance."""
    eq_cols = [c for c in data.columns if "IsEquals" in c or "IsEqual" in c]
    similarity_stats = []
    
    # Échantillonnage intelligent pour de gros datasets
    sample_size = min(2000, len(data))  # Limite à 2000 pour la performance
    if len(data) > sample_size:
        sample_data = data.sample(n=sample_size, random_state=42)
        st.info(f"📊 Analyse de similarité sur un échantillon de {sample_size:,} lignes pour optimiser les performances")
    else:
        sample_data = data
    
    for col in eq_cols:
        base_field = col.replace("_IsEquals", "").replace("_IsEqual", "")
        crm_col = f"{base_field}_CRM"
        bi_col = f"{base_field}_BI"
        
        if crm_col in data.columns and bi_col in data.columns:
            # Optimisation : ne calculer que pour les valeurs non-nulles
            valid_mask = sample_data[crm_col].notna() & sample_data[bi_col].notna()
            valid_data = sample_data[valid_mask]
            
            if len(valid_data) > 0:
                # Vectorisation partielle avec apply optimisé
                similarities = valid_data.apply(
                    lambda row: calculate_similarity_cached(
                        str(row[crm_col])[:100],  # Limiter la longueur pour la performance
                        str(row[bi_col])[:100]
                    ), axis=1
                )
                
                display_name = get_field_display_name(col)
                
                similarity_stats.append({
                    'field': display_name,
                    'avg_similarity': round(similarities.mean(), 3),
                    'min_similarity': round(similarities.min(), 3),
                    'max_similarity': round(similarities.max(), 3),
                    'similarity_std': round(similarities.std(), 3)
                })
    
    return pd.DataFrame(similarity_stats)

def create_quality_metrics_dashboard(field_stats_df, overall_stats):
    """Create quality metrics dashboard - optimized."""
    st.markdown("## 📊 Métriques de qualité globales")
    
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
    
    # Graphique optimisé
    pie_data = {
        'status': ['Correspondances', 'Écarts', 'Valeurs nulles'],
        'count': [overall_stats['total_match'], overall_stats['total_no_match'], overall_stats['total_both_null']]
    }
    
    colors = ["#7fbfdc", "#6ba6b6", "#4cadb4"]
    
    fig_pie = px.pie(
        values=pie_data['count'], 
        names=pie_data['status'],
        title="🥧 Répartition globale de la qualité des données",
        color_discrete_sequence=colors
    )
    fig_pie.update_traces(hole=0.4)
    fig_pie.update_layout(showlegend=True, height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

def create_field_analysis_charts(field_stats_df):
    """Create detailed field analysis charts - optimized."""
    st.markdown("## 📈 Analyse détaillée par champ")
    
    if field_stats_df.empty:
        st.warning("Aucune donnée d'analyse de champ disponible")
        return
    
    # Graphique 1 : optimisé
    df_sorted = field_stats_df.sort_values('match_rate')
    
    fig_match_rate = px.bar(
        df_sorted,
        x='match_rate',
        y='field',
        orientation='h',
        title="📊 Taux de correspondance par champ (%)",
        color='match_rate',
        color_continuous_scale=[[0, "#6ba6b6"], [0.5, "#4cadb4"], [1, "#7fbfdc"]],
        text='match_rate'
    )
    fig_match_rate.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_match_rate.update_layout(height=max(400, len(field_stats_df) * 25), showlegend=False)
    st.plotly_chart(fig_match_rate, use_container_width=True)
    
    # Graphique 2 : optimisé avec melt
    status_df = field_stats_df.melt(
        id_vars=['field'], 
        value_vars=['match', 'no_match', 'both_null'],
        var_name='status', 
        value_name='count'
    )
    
    # Mapper les noms
    status_mapping = {
        'match': 'Correspondances',
        'no_match': 'Écarts', 
        'both_null': 'Valeurs nulles'
    }
    status_df['status'] = status_df['status'].map(status_mapping)
    
    color_map = {
        'Correspondances': '#7fbfdc',
        'Écarts': '#6ba6b6',
        'Valeurs nulles': '#4cadb4'
    }
    
    fig_stacked = px.bar(
        status_df,
        x='field',
        y='count',
        color='status',
        title="📚 Répartition détaillée par champ",
        color_discrete_map=color_map
    )
    fig_stacked.update_xaxes(tickangle=45)
    fig_stacked.update_layout(height=500)
    st.plotly_chart(fig_stacked, use_container_width=True)

def create_similarity_analysis(similarity_stats_df):
    """Create similarity analysis charts - optimized."""
    if similarity_stats_df.empty:
        return
        
    st.markdown("## 🔍 Analyse de similarité textuelle")
    
    df_sorted = similarity_stats_df.sort_values('avg_similarity')
    
    fig_sim = px.bar(
        df_sorted,
        x='avg_similarity',
        y='field',
        orientation='h',
        title="📝 Score de similarité moyen par champ",
        color='avg_similarity',
        color_continuous_scale=[[0, "#6ba6b6"], [0.5, "#78b495"], [1, "#82b86a"]],
        text='avg_similarity'
    )
    fig_sim.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_sim.update_layout(height=max(400, len(similarity_stats_df) * 25), showlegend=False)
    st.plotly_chart(fig_sim, use_container_width=True)
    
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
        },
        color_discrete_sequence=["#45b49d"]
    )
    fig_sim_dist.update_layout(height=400)
    st.plotly_chart(fig_sim_dist, use_container_width=True)

def create_advanced_analytics(df, field_stats_df):
    """Create advanced analytics - optimized."""
    st.markdown("## 🎯 Analyses avancées")
    
    if field_stats_df.empty:
        return
    
    # Calculs optimisés
    df_copy = field_stats_df.copy()
    df_copy['total_records'] = df_copy['match'] + df_copy['no_match'] + df_copy['both_null']
    df_copy['error_rate'] = np.where(df_copy['total_records'] > 0, 
                                    (df_copy['no_match'] / df_copy['total_records'] * 100), 0)
    
    # Simulation rapide de complexité
    np.random.seed(42)  # Pour la reproductibilité
    df_copy['fix_complexity'] = np.random.randint(1, 5, size=len(df_copy))
    df_copy['business_impact'] = df_copy['error_rate'] / 20
    
    fig_priority = px.scatter(
        df_copy,
        x='fix_complexity',
        y='business_impact',
        size='no_match',
        hover_data=['field', 'error_rate'],
        title="🎯 Matrice de priorisation (Impact métier vs Complexité de correction)",
        labels={
            'fix_complexity': 'Complexité de correction (1=facile, 5=difficile)',
            'business_impact': 'Impact métier'
        },
        color_discrete_sequence=["#7fbfdc"]
    )
    
    # Lignes de référence
    median_impact = df_copy['business_impact'].median()
    median_complexity = df_copy['fix_complexity'].median()
    
    fig_priority.add_hline(y=median_impact, line_dash="dash", line_color="gray")
    fig_priority.add_vline(x=median_complexity, line_dash="dash", line_color="gray")
    
    # Annotations
    fig_priority.add_annotation(x=1.2, y=4.5, text="🚨 URGENT", bgcolor="rgba(255,0,0,0.1)")
    fig_priority.add_annotation(x=4.8, y=4.5, text="🎯 PLANIFIER", bgcolor="rgba(255,165,0,0.1)")
    fig_priority.add_annotation(x=1.2, y=0.5, text="✅ QUICK WINS", bgcolor="rgba(0,255,0,0.1)")
    fig_priority.add_annotation(x=4.8, y=0.5, text="⏳ REPORTER", bgcolor="rgba(128,128,128,0.1)")
    
    fig_priority.update_layout(height=500)
    st.plotly_chart(fig_priority, use_container_width=True)

# App Header
st.markdown("""
# 🎯 Analyse de Qualité des Données CRM vs BI
### Diagnostic complet pour l'amélioration de la correspondance des données
""")

# Sidebar avec bouton Appliquer
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
        show_quality_metrics = st.checkbox("📊 Métriques de qualité globales", value=True)
        show_field_analysis = st.checkbox("📈 Analyse détaillée par champ", value=True)
        show_similarity = st.checkbox("🔍 Analyse de similarité", value=False,
                                    help="⚠️ Peut ralentir l'analyse pour de gros fichiers")
        show_advanced = st.checkbox("🎯 Analyses avancées", value=True)
        
        st.markdown("---")
        
        # Bouton Appliquer avec style
        apply_analysis = st.button(
            "🚀 Appliquer l'analyse", 
            type="primary",
            use_container_width=True,
            help="Lancer l'analyse avec les options sélectionnées"
        )
        
        # Initialiser la session state
        if 'analysis_applied' not in st.session_state:
            st.session_state.analysis_applied = False
        
        if apply_analysis:
            st.session_state.analysis_applied = True
            st.session_state.show_quality_metrics = show_quality_metrics
            st.session_state.show_field_analysis = show_field_analysis
            st.session_state.show_similarity = show_similarity
            st.session_state.show_advanced = show_advanced

# Main App Logic
if uploaded_file is not None and st.session_state.get('analysis_applied', False):
    try:
        # Optimisation : lire le fichier une seule fois
        if 'df_loaded' not in st.session_state or st.session_state.get('last_file_name') != uploaded_file.name:
            with st.spinner("🔄 Chargement du fichier..."):
                file_bytes = uploaded_file.read()
                uploaded_file.seek(0)  # Reset pour les prochaines lectures
                df = load_accounts_file(file_bytes, uploaded_file.name)
                st.session_state.df_loaded = df
                st.session_state.last_file_name = uploaded_file.name
        else:
            df = st.session_state.df_loaded
        
        # Analyse principale rapide
        with st.spinner("⚡ Analyse en cours..."):
            overall_stats, row_stats, field_stats_df, _ = compute_comprehensive_analysis(df)
        
        # Métriques de base
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total lignes", f"{len(df):,}")
        with col2:
            eq_cols = [c for c in df.columns if "IsEquals" in c]
            st.metric("🔗 Champs comparés", len(eq_cols))
        with col3:
            st.metric("🚨 Lignes avec au moins 1 écart", f"{row_stats.get('rows_with_nomatch', 0):,}")
        with col4:
            st.metric("✅ Lignes ISO", f"{row_stats.get('rows_iso', 0):,}")
        
        # Analyses conditionnelles basées sur les options
        if st.session_state.get('show_quality_metrics', True) and overall_stats:
            create_quality_metrics_dashboard(field_stats_df, overall_stats)
        
        if st.session_state.get('show_field_analysis', True) and not field_stats_df.empty:
            create_field_analysis_charts(field_stats_df)
        
        if st.session_state.get('show_similarity', False) and not field_stats_df.empty:
            with st.spinner("🔍 Calcul de la similarité..."):
                similarity_stats_df = compute_similarity_analysis(df, field_stats_df)
                if not similarity_stats_df.empty:
                    create_similarity_analysis(similarity_stats_df)
        
        if st.session_state.get('show_advanced', True) and not field_stats_df.empty:
            create_advanced_analytics(df, field_stats_df)
        
        # Aperçu des données
        st.markdown("## 👀 Aperçu des données")
        
        with st.expander("📊 Statistiques résumées"):
            st.dataframe(field_stats_df, use_container_width=True)
        
        with st.expander("🔍 Échantillon des données brutes"):
            st.dataframe(df.head(20), use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement du fichier: {str(e)}")
        st.info("💡 Vérifiez le format de votre fichier et réessayez.")
        
        with st.expander("🔍 Informations de débogage"):
            st.code(f"Type d'erreur: {type(e).__name__}")
            st.code(f"Message: {str(e)}")

elif uploaded_file is not None and not st.session_state.get('analysis_applied', False):
    st.info("👆 Configurez vos options d'analyse dans la barre latérale et cliquez sur **'🚀 Appliquer l'analyse'** pour commencer.")

else:
    # Welcome screen
    st.markdown("""
    ## 👋 Bienvenue dans l'outil d'analyse de qualité des données
    
    Cet outil vous permet d'analyser la qualité de correspondance entre vos données CRM et BI.
    
    ### 🚀 Pour commencer:
    1. **📁 Uploadez votre fichier** dans la barre latérale (Excel ou CSV)
    2. **⚙️ Configurez les options** d'analyse selon vos besoins
    3. **🚀 Cliquez sur "Appliquer l'analyse"** pour lancer le traitement
    4. **📊 Explorez les résultats** avec les graphiques interactifs
    
    ### ⚡ Optimisations de performance:
    - 🔄 **Mise en cache intelligente** des données

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
    st.dataframe(sample_data, use_container_width=True)