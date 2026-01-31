import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, 
    precision_recall_curve, classification_report
)
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go

# ========================
# CONFIGURATION & STYLE
# ========================
st.set_page_config(page_title="User Behavior SVM Pro", layout="wide")

# --- STYLE CSS POUR LES CARTES ---
st.markdown("""
<style>
    /* Style de la boîte globale de la métrique */
    [data-testid="stMetric"] {
        background-color: #f8f9fb; /* Gris très clair pour le fond */
        border: 1px solid #e0e0e0; /* Bordure légère */
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05); /* Ombre portée douce */
        transition: transform 0.3s ease;
    }
    
    /* Effet au survol */
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #1E3A8A; /* La bordure devient bleue au survol */
    }

    /* Couleur du titre de la carte */
    [data-testid="stMetricLabel"] p {
        color: #555555 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }

    /* Couleur de la valeur principale */
    [data-testid="stMetricValue"] div {
        color: #1E3A8A !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CHARGEMENT DES DONNÉES ET DU MODÈLE ---
@st.cache_data
def load_data():
    return pd.read_csv('realistic_user_behavior_dataset_1000.csv')

def load_model():
    with open('model_final.pkl', 'rb') as f:
        return pickle.load(f)

df = load_data()
saved_data = load_model()
model = saved_data['model']
scaler = saved_data['scaler']
features = saved_data['features']

# ========================
# SIDEBAR DYNAMIQUE
# ========================
st.sidebar.title("🤖 Navigation")
page = st.sidebar.radio("Aller vers :", ["📊 Exploration & Stratégie", "🔮 Prédiction Interactive"])

st.sidebar.divider()
st.sidebar.markdown("### 📊 État du Dataset")
st.sidebar.info(f"""
- **Total lignes**: {len(df)}
- **Variables**: {len(features)}
- **Déséquilibre**: {round(df['target'].mean()*100, 1)}% de cibles
""")

# ========================
# PAGE 1 : EXPLORATION
# ========================
if page == "📊 Exploration & Stratégie":
    st.markdown('<h1 class="main-header">Tableau de Bord Stratégique SVM</h1>', unsafe_allow_html=True)
    
    # --- CALCULS KPI ---
    X_all_sc = scaler.transform(df[features])
    all_probs = model.predict_proba(X_all_sc)[:, 1]
    gold_opportunities = ((all_probs > 0.7) & (df['target'] == 0)).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Base Utilisateurs", len(df))
    col2.metric("Taux de Conversion", f"{round(df['target'].mean()*100, 1)}%", delta="-2.1% vs obj")
    col3.metric("Opportunités Gold", gold_opportunities, delta="Priorité Haute", delta_color="inverse")
    col4.metric("Revenu Moyen", f"{round(df['income'].mean(), 0)} €")

    st.divider()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔥 Corrélations", "📈 Distributions", "🎯 Performance Modèle", "🏆 Segmentation", "🧠 Insight"
    ])   

    with tab1:
        st.subheader("Analyse des corrélations numériques")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='RdYlGn', fmt=".2f", ax=ax)
        st.pyplot(fig)

    with tab2:
        st.subheader("Distribution par segment de clientèle")
        var_to_plot = st.selectbox("Feature à analyser :", features)
        fig = px.histogram(df, x=var_to_plot, color="target", marginal="box", 
                           barmode="overlay", color_discrete_map={0: '#EF4444', 1: '#10B981'})
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Fiabilité & Courbes de Décision")
        y_pred_all = model.predict(X_all_sc)
        cm = confusion_matrix(df['target'], y_pred_all)
        
        c1, c2 = st.columns(2)
        with c1:
            # Matrice de confusion interactive
            fig_cm = ff.create_annotated_heatmap(cm, x=['Prédit: 0', 'Prédit: 1'], y=['Réel: 0', 'Réel: 1'], colorscale='Blues')
            st.plotly_chart(fig_cm, use_container_width=True)
        with c2:
            # Courbe ROC
            fpr, tpr, _ = roc_curve(df['target'], all_probs)
            fig_roc = px.area(x=fpr, y=tpr, title=f'Courbe ROC (AUC={auc(fpr, tpr):.3f})',
                              labels=dict(x='Taux Faux Positifs', y='Taux Vrais Positifs'))
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig_roc, use_container_width=True)

    with tab4:
        st.subheader("Segmentation par Score de Probabilité")
        temp_df = df.copy()
        temp_df['Probability'] = all_probs
        temp_df['Segment'] = pd.cut(temp_df['Probability'], bins=[0, 0.3, 0.7, 1], labels=['Bronze', 'Silver', 'Gold'])
        
        fig_seg = px.treemap(temp_df, path=['Segment'], values='income', color='Segment',
                             color_discrete_map={'Bronze':'#cd7f32', 'Silver':'#c0c0c0', 'Gold':'#ffd700'})
        st.plotly_chart(fig_seg, use_container_width=True)

    with tab5:
        st.subheader("Nuage de points Multivarié")
        fig_px = px.scatter(df, x=features[0], y=features[1], color="target", size="income",
                            hover_data=features, color_continuous_scale="Viridis")
        st.plotly_chart(fig_px, use_container_width=True)

# ========================
# PAGE 2 : PRÉDICTION
# ========================
else:
    st.markdown('<h1 class="main-header">🔮 Simulateur de Prédiction SVM</h1>', unsafe_allow_html=True)
    
    with st.expander("Ajuster les paramètres de l'utilisateur", expanded=True):
        input_data = {}
        cols = st.columns(4)
        for i, feature in enumerate(features):
            with cols[i % 4]:
                # Utilisation de sliders pour une meilleure expérience
                input_data[feature] = st.slider(
                    f"{feature}", 
                    float(df[feature].min()), 
                    float(df[feature].max()), 
                    float(df[feature].mean())
                )

    if st.button("Lancer l'Analyse du Profil"):
        input_df = pd.DataFrame([input_data])
        input_sc = scaler.transform(input_df)
        prediction = model.predict(input_sc)[0]
        probability = model.predict_proba(input_sc)[0]

        st.divider()
        
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            st.subheader("Résultat ")
            if prediction == 1:
                st.success(f"### CLASSE 1 (CIBLE)\nProbabilité : {probability[1]:.1%}")
                st.balloons()
            else:
                st.error(f"### CLASSE 0 (NON-CIBLE)\nProbabilité : {probability[0]:.1%}")
            
            # Gauge de probabilité
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability[1] * 100,
                title = {'text': "Score de Conversion"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#1E3A8A"}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with c2:
            st.subheader("Analyse Radar du Profil")
            mean_target_1 = df[df['target'] == 1][features].mean()
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=[input_data[f] for f in features], theta=features, fill='toself', name='Profil actuel'))
            fig_radar.add_trace(go.Scatterpolar(r=[mean_target_1[f] for f in features], theta=features, fill='toself', name='Moyenne Cibles'))
            
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
            st.plotly_chart(fig_radar, use_container_width=True)