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
    if file_name.endswith(".csv"):
        try:
            try:
                content = uploaded_file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                content = uploaded_file_bytes.decode('iso-8859-1')
            
            first_line = content.split('\n')[0]
            delimiter = ',' if ',' in first_line else ';' if ';' in first_line else '\t'
            
            from io import StringIO
            df = pd.read_csv(StringIO(content), sep=delimiter, engine="c", low_memory=False)
        except Exception:
            from io import BytesIO
            df = pd.read_csv(BytesIO(uploaded_file_bytes), engine="python", on_bad_lines="skip")
    else:
        from io import BytesIO
        df = pd.read_excel(BytesIO(uploaded_file_bytes), engine='openpyxl')
    
    return df

def normalize_bool(val):
    """Return True if val looks like a positive boolean."""
    if pd.isna(val):
        return False
    val_str = str(val).strip().lower()
    return val_str in {"true", "1", "yes", "y", "t", "match"}

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
    """Compute comprehensive analysis of data quality - basé sur la version fonctionnelle."""
    eq_cols = [c for c in data.columns if "IsEquals" in c or "IsEqual" in c]
    
    if not eq_cols:
        return {}, {}, pd.DataFrame(), pd.DataFrame()
    
    total_rows = len(data)
    field_stats = []
    detailed_stats = []
    
    # Calcul des lignes avec écarts (basé sur la version fonctionnelle)
    rows_with_nomatch = 0
    if eq_cols:
        has_nomatch_per_row = pd.Series(False, index=data.index)
        
        for col in eq_cols:
            # Lecture directe des valeurs "No Match"
            col_values = data[col].astype(str).str.strip().str.lower()
            no_match_series = col_values.isin({"no match", "nomatch", "no_match", "false", "0", "no", "n", "f"})
            has_nomatch_per_row |= no_match_series
        
        rows_with_nomatch = int(has_nomatch_per_row.sum())
    
    rows_iso = total_rows - rows_with_nomatch
    
    # Analyse chaque champ de comparaison avec la nouvelle logique
    for col in eq_cols:
        # Trouver l'index de la colonne IsEquals
        col_index = data.columns.get_loc(col)
        
        # Vérifier qu'il y a au moins 2 colonnes avant
        if col_index < 2:
            continue
            
        # Récupérer les 2 colonnes précédentes
        col_before_1 = data.columns[col_index - 2]  # 2 colonnes avant
        col_before_2 = data.columns[col_index - 1]  # 1 colonne avant
        
        # Identifier quelle colonne est CRM et laquelle est BI
        if "_CRM" in col_before_1 and "_BI" in col_before_2:
            crm_col = col_before_1
            bi_col = col_before_2
        elif "_BI" in col_before_1 and "_CRM" in col_before_2:
            crm_col = col_before_2
            bi_col = col_before_1
        else:
            # Si on ne peut pas identifier CRM/BI, passer au suivant
            continue
        
        # DEBUG - Afficher les informations
        st.write(f"🔍 **Analyse de {col}:**")
        st.write(f"- Position: colonne {col_index}")
        st.write(f"- Colonne CRM: {crm_col}")
        st.write(f"- Colonne BI: {bi_col}")
        
        # Calculs de base (logique existante)
        match_series = data[col].apply(normalize_bool)
        match_count = int(match_series.sum())
        
        both_null_series = data[col].astype(str).str.strip().str.lower().isin(['both null', 'bothnull', 'null', 'both_null'])
        both_null_count = int(both_null_series.sum())
        
        no_match_count = total_rows - match_count - both_null_count
        
        # Analyse détaillée des cas No Match avec la nouvelle logique
        crm_null_bi_value = 0
        bi_null_crm_value = 0
        different_values = 0
        
        if no_match_count > 0:
            # Filtrer les lignes avec No Match
            no_match_mask = ~match_series & ~both_null_series
            no_match_data = data[no_match_mask]
            
            st.write(f"- Lignes No Match: {len(no_match_data)}")
            
            if len(no_match_data) > 0:
                # Montrer un échantillon
                st.write("**Échantillon des données No Match:**")
                sample_cols = [crm_col, bi_col, col]
                st.dataframe(no_match_data[sample_cols].head(10))
                
                # Analyser chaque ligne No Match
                for idx, row in no_match_data.iterrows():
                    crm_val = row[crm_col]
                    bi_val = row[bi_col]
                    
                    # CRM vide/null mais BI a une valeur
                    if (pd.isna(crm_val) or str(crm_val).strip() == '') and (pd.notna(bi_val) and str(bi_val).strip() != ''):
                        crm_null_bi_value += 1
                    # BI vide/null mais CRM a une valeur
                    elif (pd.isna(bi_val) or str(bi_val).strip() == '') and (pd.notna(crm_val) and str(crm_val).strip() != ''):
                        bi_null_crm_value += 1
                    # Les deux ont des valeurs différentes (non vides)
                    elif (pd.notna(crm_val) and str(crm_val).strip() != '') and (pd.notna(bi_val) and str(bi_val).strip() != ''):
                        different_values += 1
                    else:
                        different_values += 1
                
                st.write(f"**Résultats:**")
                st.write(f"- CRM null / BI valeur: {crm_null_bi_value}")
                st.write(f"- BI null / CRM valeur: {bi_null_crm_value}")
                st.write(f"- Valeurs différentes: {different_values}")
                st.write("---")
        
        match_rate = (match_count / total_rows * 100) if total_rows > 0 else 0
        display_name = get_field_display_name(col)
        
        field_stats.append({
            'field': display_name,
            'match': match_count,
            'no_match': no_match_count,
            'both_null': both_null_count,
            'match_rate': round(match_rate, 2)
        })
        
        detailed_stats.append({
            'field': display_name,
            'match': match_count,
            'both_null': both_null_count,
            'crm_null_bi_value': crm_null_bi_value,
            'bi_null_crm_value': bi_null_crm_value,
            'different_values': different_values
        })
    
    field_stats_df = pd.DataFrame(field_stats)
    detailed_stats_df = pd.DataFrame(detailed_stats)
    
    # Calcul global avec séparation des types de No Match
    total_match = int(field_stats_df['match'].sum()) if not field_stats_df.empty else 0
    total_both_null = int(field_stats_df['both_null'].sum()) if not field_stats_df.empty else 0
    total_crm_null = int(detailed_stats_df['crm_null_bi_value'].sum()) if not detailed_stats_df.empty else 0
    total_bi_null = int(detailed_stats_df['bi_null_crm_value'].sum()) if not detailed_stats_df.empty else 0
    total_different = int(detailed_stats_df['different_values'].sum()) if not detailed_stats_df.empty else 0
    
    overall_stats = {
        'total_match': total_match,
        'total_both_null': total_both_null,
        'total_crm_null': total_crm_null,
        'total_bi_null': total_bi_null,
        'total_different': total_different,
    }
    
    row_stats = {
        'rows_with_nomatch': rows_with_nomatch,
        'rows_iso': rows_iso
    }
    
    return overall_stats, row_stats, field_stats_df, detailed_stats_df

def create_quality_metrics_dashboard(field_stats_df, overall_stats):
    """Create quality metrics dashboard with detailed breakdown."""
    st.markdown("## 📊 Métriques de qualité globales")
    
    # Vérifier que toutes les clés existent avec des valeurs par défaut
    total_match = overall_stats.get('total_match', 0)
    total_both_null = overall_stats.get('total_both_null', 0)
    total_crm_null = overall_stats.get('total_crm_null', 0)
    total_bi_null = overall_stats.get('total_bi_null', 0)
    total_different = overall_stats.get('total_different', 0)
    
    total_comparisons = total_match + total_both_null + total_crm_null + total_bi_null + total_different
    match_rate = (total_match / total_comparisons * 100) if total_comparisons > 0 else 0
    
    # Première ligne de métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Taux de correspondance global", f"{match_rate:.1f}%")
    with col2:
        st.metric("✅ Total correspondances", f"{total_match:,}")
    with col3:
        st.metric("⚪ Valeurs nulles (deux côtés)", f"{total_both_null:,}")
    with col4:
        st.metric("🔄 Valeurs différentes", f"{total_different:,}")
    
    # Deuxième ligne de métriques pour les cas null
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("📤 CRM null / BI valeur", f"{total_crm_null:,}")
    with col6:
        st.metric("📥 BI null / CRM valeur", f"{total_bi_null:,}")
    with col7:
        total_no_match = total_crm_null + total_bi_null + total_different
        st.metric("❌ Total écarts", f"{total_no_match:,}")
    with col8:
        st.metric("📊 Total comparaisons", f"{total_comparisons:,}")
    
    # Graphique détaillé
    pie_data = pd.DataFrame([
        {'status': 'Correspondances', 'count': total_match},
        {'status': 'Valeurs nulles (deux côtés)', 'count': total_both_null},
        {'status': 'CRM null / BI valeur', 'count': total_crm_null},
        {'status': 'BI null / CRM valeur', 'count': total_bi_null},
        {'status': 'Valeurs différentes', 'count': total_different}
    ])
    
    colors = ["#7fbfdc", "#4cadb4", "#6ba6b6", "#78b495", "#82b86a"]
    
    fig_pie = px.pie(
        pie_data, 
        names='status', 
        values='count',
        title="🥧 Répartition détaillée de la qualité des données",
        color_discrete_sequence=colors
    )
    fig_pie.update_traces(hole=0.4)
    fig_pie.update_layout(showlegend=True, height=500)
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

def create_advanced_analytics(df, field_stats_df, detailed_stats_df):
    """Create advanced analytics with business rules."""
    st.markdown("## 🎯 Analyses avancées")
    
    if field_stats_df.empty or detailed_stats_df.empty:
        st.warning("Pas de données pour l'analyse avancée")
        return
    
    # Graphique de répartition détaillée par type
    st.markdown("### 📊 Répartition détaillée par champ et type")
    
    try:
        # Préparer les données pour le graphique empilé
        detailed_data = []
        for _, row in detailed_stats_df.iterrows():
            detailed_data.extend([
                {'field': row['field'], 'type': 'Correspondances', 'count': row['match']},
                {'field': row['field'], 'type': 'Valeurs nulles (deux côtés)', 'count': row['both_null']},
                {'field': row['field'], 'type': 'CRM null / BI valeur', 'count': row['crm_null_bi_value']},
                {'field': row['field'], 'type': 'BI null / CRM valeur', 'count': row['bi_null_crm_value']},
                {'field': row['field'], 'type': 'Valeurs différentes', 'count': row['different_values']}
            ])
        
        detailed_df = pd.DataFrame(detailed_data)
        
        color_map = {
            'Correspondances': '#7fbfdc',
            'Valeurs nulles (deux côtés)': '#4cadb4',
            'CRM null / BI valeur': '#6ba6b6',
            'BI null / CRM valeur': '#78b495',
            'Valeurs différentes': '#82b86a'
        }
        
        fig_detailed = px.bar(
            detailed_df,
            x='field',
            y='count',
            color='type',
            title="📚 Répartition détaillée par champ et type de situation",
            color_discrete_map=color_map
        )
        fig_detailed.update_xaxes(tickangle=45)
        fig_detailed.update_layout(height=600)
        st.plotly_chart(fig_detailed, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur dans le graphique détaillé: {str(e)}")
    
    # Matrice de priorisation avec règles business
    st.markdown("### 🎯 Matrice de priorisation")
    
    try:
        # Calculs avec règles business
        df_priority = detailed_stats_df.copy()
        df_priority['total_records'] = (df_priority['match'] + df_priority['both_null'] + 
                                      df_priority['crm_null_bi_value'] + df_priority['bi_null_crm_value'] + 
                                      df_priority['different_values'])
        
        # Éviter la division par zéro
        df_priority['total_records'] = df_priority['total_records'].replace(0, 1)
        
        # Règle 1: Plus de Both Null = Plus de complexité
        df_priority['both_null_rate'] = df_priority['both_null'] / df_priority['total_records']
        
        # Règle 2: Plus de valeurs différentes = Plus d'impact métier  
        df_priority['different_values_rate'] = df_priority['different_values'] / df_priority['total_records']
        
        # Calcul de la complexité (1-5) basé sur le taux de Both Null
        df_priority['fix_complexity'] = np.where(
            df_priority['both_null_rate'] == 0, 1,  # Pas de Both Null = facile
            np.where(df_priority['both_null_rate'] <= 0.1, 2,  # Peu de Both Null = assez facile
            np.where(df_priority['both_null_rate'] <= 0.3, 3,  # Moyen
            np.where(df_priority['both_null_rate'] <= 0.6, 4, 5)))  # Beaucoup = difficile
        )
        
        # Calcul de l'impact métier (0-5) basé sur le taux de valeurs différentes
        df_priority['business_impact'] = df_priority['different_values_rate'] * 5
        
        fig_priority = px.scatter(
            df_priority,
            x='fix_complexity',
            y='business_impact',
            size='different_values',
            hover_data=['field', 'both_null_rate', 'different_values_rate'],
            title="🎯 Matrice de priorisation (Impact métier vs Complexité de correction)",
            labels={
                'fix_complexity': 'Complexité de correction (1=facile, 5=difficile)',
                'business_impact': 'Impact métier (basé sur valeurs différentes)'
            },
            color='business_impact',
            color_continuous_scale=[[0, "#7fbfdc"], [0.5, "#78b495"], [1, "#82b86a"]]
        )
        
        # Lignes de référence
        fig_priority.add_hline(y=2.5, line_dash="dash", line_color="gray", annotation_text="Impact moyen")
        fig_priority.add_vline(x=3, line_dash="dash", line_color="gray", annotation_text="Complexité moyenne")
        
        # Annotations des quadrants
        fig_priority.add_annotation(x=1.5, y=4, text="🚨 PRIORITÉ MAX<br>(Facile + Impact élevé)", 
                                   bgcolor="rgba(255,0,0,0.1)", bordercolor="red")
        fig_priority.add_annotation(x=4.5, y=4, text="🎯 PLANIFIER<br>(Difficile + Impact élevé)", 
                                   bgcolor="rgba(255,165,0,0.1)", bordercolor="orange")
        fig_priority.add_annotation(x=1.5, y=1, text="✅ QUICK WINS<br>(Facile + Impact faible)", 
                                   bgcolor="rgba(0,255,0,0.1)", bordercolor="green")
        fig_priority.add_annotation(x=4.5, y=1, text="⏳ REPORTER<br>(Difficile + Impact faible)", 
                                   bgcolor="rgba(128,128,128,0.1)", bordercolor="gray")
        
        fig_priority.update_layout(height=600)
        st.plotly_chart(fig_priority, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur dans la matrice de priorisation: {str(e)}")

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
            overall_stats, row_stats, field_stats_df, detailed_stats_df = compute_comprehensive_analysis(df)
        
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
        
        if st.session_state.get('show_advanced', True) and not field_stats_df.empty:
            create_advanced_analytics(df, field_stats_df, detailed_stats_df)
        
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
    - 📊 **Calculs vectorisés** pour une analyse rapide
    - 🎯 **Analyses à la demande** selon vos besoins
    - 📈 **Analyse détaillée des écarts** par type
    
    ### 🎯 Fonctionnalités principales:
    - ✅ **Analyse de correspondance** par champ
    - 🎯 **Matrice de priorisation** intelligente
    - 📊 **Métriques de qualité** détaillées
    - 🔍 **Analyse des types d'écarts** (null vs différent)
    
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
    st.dataframe(sample_data, use_container_width=True)