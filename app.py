import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

# Configuration
st.set_page_config(page_title="User Behavior Dashboard", layout="wide")

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

# --- SIDEBAR NAV ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers :", ["Exploration des données", "Prédiction SVM"])

# --- PAGE 1 : EXPLORATION ---
if page == "Exploration des données":
    st.title("📊 Analyse du Comportement Utilisateur")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Utilisateurs", len(df))
    col2.metric("Taux de Conversion", f"{round(df['target'].mean()*100, 2)}%")
    col3.metric("Revenu Moyen", f"{round(df['income'].mean(), 0)} €")

    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Corrélations", "📈 Distributions", "🎯 Fiabilité Business", "🏆 Segmentation"])   

    with tab1:
        st.subheader("Quelles variables influencent la cible ?")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', fmt=".2f", ax=ax)
        st.pyplot(fig)

    with tab2:
        st.subheader("Analyse comparative par Target")
        var_to_plot = st.selectbox("Choisir une variable à analyser :", df.columns[:-1])
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, x=var_to_plot, hue="target", fill=True, palette="viridis", ax=ax)
        st.pyplot(fig)

    with tab3:
        st.subheader("Interprétation de la Matrice de Confusion")
        st.write("Le modèle est-il vraiment utile pour l'entreprise ?")
        
        # Calcul de la matrice sur les données actuelles (pour l'exemple)
        X_all = scaler.transform(df[features])
        y_pred_all = model.predict(X_all)
        cm = confusion_matrix(df['target'], y_pred_all)

        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            st.markdown(f"""
            **Bilan des prédictions :**
            - ✅ **Succès :** {cm[1,1]} clients correctement ciblés.
            - 🟠 **Fausses Alertes :** {cm[0,1]} personnes ciblées pour rien.
            - ❌ **Occasions Manquées :** {cm[1,0]} clients potentiels oubliés.
            - 🛡️ **Tri Correct :** {cm[0,0]} non-clients évités.
            """)
            
            st.info("**Note métier :** Un bon modèle doit maximiser les 'Succès' tout en gardant les 'Occasions Manquées' le plus bas possible.")

        with c2:
            # Matrice version "Expert" mais lisible
            z = cm
            x = ['Prédit: NON', 'Prédit: OUI']
            y = ['Réalité: NON', 'Réalité: OUI']
            fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')
            st.plotly_chart(fig_cm, use_container_width=True)
    with tab4:
        st.subheader("Segmentation Stratégique des Utilisateurs")
    
        # Calcul des probabilités pour tout le dataset
        X_all_sc = scaler.transform(df[features])
        probs = model.predict_proba(X_all_sc)[:, 1]
        
        # Création des segments
        df['Probability'] = probs
        df['Segment'] = pd.cut(df['Probability'], 
                            bins=[0, 0.3, 0.7, 1], 
                            labels=['Bronze (Faible)', 'Silver (Moyen)', 'Gold (Prioritaire)'])

        col_s1, col_s2 = st.columns([2, 1])
        
        with col_s1:
            # Graphique de répartition des segments
            fig_seg, ax_seg = plt.subplots()
            sns.countplot(data=df, x='Segment', palette=['#cd7f32', '#c0c0c0', '#ffd700'], ax=ax_seg)
            st.pyplot(fig_seg)
            
        with col_s2:
            st.write("**Répartition des profils :**")
            segment_counts = df['Segment'].value_counts()
            for seg, count in segment_counts.items():
                st.metric(seg, f"{count} utilisateurs")

        st.divider()
        st.write("🔍 **Liste des 10 profils 'Gold' à cibler immédiatement :**")
        # On affiche les meilleurs profils qui n'ont pas encore converti (target == 0)
        potential_gold = df[(df['Segment'] == 'Gold (Prioritaire)') & (df['target'] == 0)]
        st.dataframe(potential_gold.sort_values(by='Probability', ascending=False).head(10))

# --- PAGE 2 : PRÉDICTION ---
else:
    st.title("🔮 Prédiction avec le SVM Optimisé")
    st.write("Entrez les paramètres pour tester le profil d'un utilisateur :")

    input_data = {}
    cols = st.columns(len(features))
    
    for i, feature in enumerate(features):
        with cols[i]:
            val_mean = float(df[feature].mean())
            input_data[feature] = st.number_input(f"{feature}", value=val_mean)

    if st.button("Lancer la Prédiction"):
        input_df = pd.DataFrame([input_data])
        input_sc = scaler.transform(input_df)
        
        prediction = model.predict(input_sc)[0]
        probability = model.predict_proba(input_sc)[0][1]

        st.divider()
        if prediction == 1:
            st.success(f"### Résultat : CIBLE (Probabilité : {round(probability*100, 1)}%)")
            st.balloons()
        else:
            st.error(f"### Résultat : NON CIBLE (Probabilité : {round(probability*100, 1)}%)")
        
        st.progress(probability)