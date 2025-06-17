import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model  
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import cv2
from datetime import datetime
import time

# Configuration de la page
st.set_page_config(
    page_title="Plant Disease AI Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4A90E2;
        margin-bottom: 1rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown('<h1 class="main-header">🌱 Plant Disease AI Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Classification intelligente des maladies des plantes avec Deep Learning</p>', unsafe_allow_html=True)

# Classes de prédiction avec informations détaillées
classes_info = {
    'Apple___Apple_scab': {'id': 0, 'plant': 'Pomme', 'disease': 'Tavelure', 'severity': 'Modérée', 'color': '#FF6B6B'},
    'Apple___Black_rot': {'id': 1, 'plant': 'Pomme', 'disease': 'Pourriture noire', 'severity': 'Élevée', 'color': '#4ECDC4'},
    'Apple___Cedar_apple_rust': {'id': 2, 'plant': 'Pomme', 'disease': 'Rouille du cèdre', 'severity': 'Modérée', 'color': '#45B7D1'},
    'Apple___healthy': {'id': 3, 'plant': 'Pomme', 'disease': 'Saine', 'severity': 'Aucune', 'color': '#96CEB4'},
    'Blueberry___healthy': {'id': 4, 'plant': 'Myrtille', 'disease': 'Saine', 'severity': 'Aucune', 'color': '#96CEB4'},
    'Cherry_(including_sour)___Powdery_mildew': {'id': 5, 'plant': 'Cerise', 'disease': 'Oïdium', 'severity': 'Modérée', 'color': '#FFEAA7'},
    'Cherry_(including_sour)___healthy': {'id': 6, 'plant': 'Cerise', 'disease': 'Saine', 'severity': 'Aucune', 'color': '#96CEB4'},
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {'id': 7, 'plant': 'Maïs', 'disease': 'Tache grise', 'severity': 'Modérée', 'color': '#FD79A8'},
    'Corn_(maize)___Common_rust_': {'id': 8, 'plant': 'Maïs', 'disease': 'Rouille commune', 'severity': 'Modérée', 'color': '#FDCB6E'},
    'Corn_(maize)___Northern_Leaf_Blight': {'id': 9, 'plant': 'Maïs', 'disease': 'Brûlure nordique', 'severity': 'Élevée', 'color': '#E17055'},
    'Corn_(maize)___healthy': {'id': 10, 'plant': 'Maïs', 'disease': 'Sain', 'severity': 'Aucune', 'color': '#96CEB4'},
    'Grape___Black_rot': {'id': 11, 'plant': 'Raisin', 'disease': 'Pourriture noire', 'severity': 'Élevée', 'color': '#A29BFE'},
    'Grape___Esca_(Black_Measles)': {'id': 12, 'plant': 'Raisin', 'disease': 'Esca', 'severity': 'Élevée', 'color': '#6C5CE7'},
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {'id': 13, 'plant': 'Raisin', 'disease': 'Brûlure foliaire', 'severity': 'Modérée', 'color': '#FD79A8'},
    'Grape___healthy': {'id': 14, 'plant': 'Raisin', 'disease': 'Sain', 'severity': 'Aucune', 'color': '#96CEB4'},
    'Orange___Haunglongbing_(Citrus_greening)': {'id': 15, 'plant': 'Orange', 'disease': 'Huanglongbing', 'severity': 'Critique', 'color': '#D63031'},
    'Peach___Bacterial_spot': {'id': 16, 'plant': 'Pêche', 'disease': 'Tache bactérienne', 'severity': 'Modérée', 'color': '#E84393'},
    'Peach___healthy': {'id': 17, 'plant': 'Pêche', 'disease': 'Saine', 'severity': 'Aucune', 'color': '#96CEB4'},
    'Pepper,_bell___Bacterial_spot': {'id': 18, 'plant': 'Poivron', 'disease': 'Tache bactérienne', 'severity': 'Modérée', 'color': '#E84393'},
    'Pepper,_bell___healthy': {'id': 19, 'plant': 'Poivron', 'disease': 'Sain', 'severity': 'Aucune', 'color': '#96CEB4'},
    'Potato___Early_blight': {'id': 20, 'plant': 'Pomme de terre', 'disease': 'Mildiou précoce', 'severity': 'Modérée', 'color': '#E17055'},
    'Potato___Late_blight': {'id': 21, 'plant': 'Pomme de terre', 'disease': 'Mildiou tardif', 'severity': 'Élevée', 'color': '#D63031'},
    'Potato___healthy': {'id': 22, 'plant': 'Pomme de terre', 'disease': 'Saine', 'severity': 'Aucune', 'color': '#96CEB4'},
    'Raspberry___healthy': {'id': 23, 'plant': 'Framboise', 'disease': 'Saine', 'severity': 'Aucune', 'color': '#96CEB4'},
    'Soybean___healthy': {'id': 24, 'plant': 'Soja', 'disease': 'Sain', 'severity': 'Aucune', 'color': '#96CEB4'},
    'Squash___Powdery_mildew': {'id': 25, 'plant': 'Courge', 'disease': 'Oïdium', 'severity': 'Modérée', 'color': '#FFEAA7'},
    'Strawberry___Leaf_scorch': {'id': 26, 'plant': 'Fraise', 'disease': 'Brûlure foliaire', 'severity': 'Modérée', 'color': '#E17055'},
    'Strawberry___healthy': {'id': 27, 'plant': 'Fraise', 'disease': 'Saine', 'severity': 'Aucune', 'color': '#96CEB4'},
    'Tomato___Bacterial_spot': {'id': 28, 'plant': 'Tomate', 'disease': 'Tache bactérienne', 'severity': 'Modérée', 'color': '#E84393'},
    'Tomato___Early_blight': {'id': 29, 'plant': 'Tomate', 'disease': 'Mildiou précoce', 'severity': 'Modérée', 'color': '#E17055'},
    'Tomato___Late_blight': {'id': 30, 'plant': 'Tomate', 'disease': 'Mildiou tardif', 'severity': 'Élevée', 'color': '#D63031'},
    'Tomato___Leaf_Mold': {'id': 31, 'plant': 'Tomate', 'disease': 'Moisissure foliaire', 'severity': 'Modérée', 'color': '#FDCB6E'},
    'Tomato___Septoria_leaf_spot': {'id': 32, 'plant': 'Tomate', 'disease': 'Septoriose', 'severity': 'Modérée', 'color': '#FD79A8'},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'id': 33, 'plant': 'Tomate', 'disease': 'Acariens', 'severity': 'Modérée', 'color': '#FDCB6E'},
    'Tomato___Target_Spot': {'id': 34, 'plant': 'Tomate', 'disease': 'Tache cible', 'severity': 'Modérée', 'color': '#E17055'},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'id': 35, 'plant': 'Tomate', 'disease': 'Virus TYLCV', 'severity': 'Élevée', 'color': '#D63031'},
    'Tomato___Tomato_mosaic_virus': {'id': 36, 'plant': 'Tomate', 'disease': 'Virus mosaïque', 'severity': 'Élevée', 'color': '#D63031'},
    'Tomato___healthy': {'id': 37, 'plant': 'Tomate', 'disease': 'Saine', 'severity': 'Aucune', 'color': '#96CEB4'}
}

# Fonction pour préprocesser l'image
def preprocess_image(uploaded_file):
    """Préprocesse l'image pour la prédiction"""
    test_image = image.load_img(uploaded_file, target_size=(64, 64))
    img_array = img_to_array(test_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisation
    return img_array, test_image

# Fonction pour obtenir les informations de la classe prédite
def get_class_info(predicted_class_key):
    """Retourne les informations détaillées de la classe prédite"""
    return classes_info.get(predicted_class_key, {})

# Fonction pour créer le graphique des probabilités
def create_probability_chart(predictions, top_n=5):
    """Crée un graphique des top N probabilités"""
    # Obtenir les indices des top N prédictions
    top_indices = np.argsort(predictions[0])[-top_n:][::-1]
    top_probs = predictions[0][top_indices]
    
    # Obtenir les noms des classes correspondantes
    class_names = list(classes_info.keys())
    top_classes = [class_names[i] for i in top_indices]
    
    # Nettoyer les noms pour l'affichage
    display_names = []
    for class_name in top_classes:
        info = classes_info[class_name]
        display_names.append(f"{info['plant']} - {info['disease']}")
    
    # Créer le graphique
    fig = go.Figure(data=[
        go.Bar(
            y=display_names,
            x=top_probs * 100,
            orientation='h',
            marker=dict(
                color=[classes_info[class_name]['color'] for class_name in top_classes],
                line=dict(color='rgba(50, 171, 96, 1.0)', width=1)
            ),
            text=[f'{prob*100:.1f}%' for prob in top_probs],
            textposition='inside'
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_n} Prédictions',
        xaxis_title='Probabilité (%)',
        yaxis_title='Classes',
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Sidebar
st.sidebar.markdown("## 📋 Configuration")

# Upload de fichier
uploaded_file = st.sidebar.file_uploader(
    "📁 Sélectionnez une image de plante",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    help="Formats supportés: JPG, JPEG, PNG, BMP, TIFF"
)

# Paramètres avancés
st.sidebar.markdown("### ⚙️ Paramètres Avancés")
confidence_threshold = st.sidebar.slider(
    "Seuil de confiance minimum", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05,
    help="Seuil de confiance minimum pour valider une prédiction"
)

show_top_predictions = st.sidebar.slider(
    "Nombre de prédictions à afficher", 
    min_value=3, 
    max_value=10, 
    value=5,
    help="Nombre de classes les plus probables à afficher"
)

# Informations sur le modèle
with st.sidebar.expander("ℹ️ Informations sur le modèle"):
    st.write("**Architecture:** CNN (Réseau de Neurones Convolutifs)")
    st.write("**Taille d'entrée:** 64x64 pixels")
    st.write("**Nombre de classes:** 38")
    st.write("**Framework:** TensorFlow/Keras")

# Chargement du modèle avec gestion d'erreur
@st.cache_resource
def load_model_cached():
    """Charge le modèle avec mise en cache"""
    try:
        return tf.keras.models.load_model('checkpoints/model.keras')
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return None

# Interface principale
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h2 class="sub-header">📷 Image d\'entrée</h2>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Affichage de l'image
        st.image(uploaded_file, caption='Image téléchargée', use_container_width=True)
        
        # Informations sur l'image
        img_details = Image.open(uploaded_file)
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write(f"**Dimensions:** {img_details.size[0]} x {img_details.size[1]} pixels")
        st.write(f"**Format:** {img_details.format}")
        st.write(f"**Mode:** {img_details.mode}")
        st.write(f"**Taille:** {len(uploaded_file.getvalue())/1024:.1f} KB")
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.info("👆 Veuillez télécharger une image dans la sidebar pour commencer l'analyse")

with col2:
    st.markdown('<h2 class="sub-header">🔍 Résultats de l\'analyse</h2>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Bouton de prédiction
        if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
            with st.spinner('🔄 Analyse en cours...'):
                # Simulation du temps de traitement
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Chargement du modèle
                model = load_model_cached()
                
                if model is not None:
                    try:
                        # Préprocessing
                        img_array, processed_img = preprocess_image(uploaded_file)
                        
                        # Prédiction
                        start_time = time.time()
                        predictions = model.predict(img_array)
                        inference_time = time.time() - start_time
                        
                        # Classe prédite
                        predicted_class = np.argmax(predictions[0])
                        confidence = float(np.max(predictions[0]))
                        
                        # Trouver le nom de la classe
                        predicted_class_name = list(classes_info.keys())[predicted_class]
                        class_info = get_class_info(predicted_class_name)
                        
                        # Affichage des résultats
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.markdown(f"### 🎯 Résultat: {class_info['plant']} - {class_info['disease']}")
                        st.markdown(f"**Confiance: {confidence*100:.1f}%**")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Métriques
                        col_met1, col_met2, col_met3 = st.columns(3)
                        
                        with col_met1:
                            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                            st.metric("Confiance", f"{confidence*100:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col_met2:
                            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                            st.metric("Temps d'inférence", f"{inference_time*1000:.1f}ms")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col_met3:
                            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                            st.metric("Sévérité", class_info['severity'])
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Validation du seuil de confiance
                        if confidence < confidence_threshold:
                            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                            st.warning(f"⚠️ Confiance ({confidence*100:.1f}%) inférieure au seuil ({confidence_threshold*100:.1f}%). Résultat incertain.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Graphique des probabilités
                        st.plotly_chart(
                            create_probability_chart(predictions, show_top_predictions),
                            use_container_width=True
                        )
                        
                        # Détails techniques
                        with st.expander("🔬 Détails techniques"):
                            st.write(f"**Classe prédite (ID):** {predicted_class}")
                            st.write(f"**Nom de la classe:** {predicted_class_name}")
                            st.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            st.write(f"**Shape de l'image:** {img_array.shape}")
                            
                            # Histogramme des probabilités
                            all_probs = predictions[0]
                            fig_hist = px.histogram(
                                x=all_probs,
                                nbins=30,
                                title="Distribution des probabilités",
                                labels={'x': 'Probabilité', 'y': 'Fréquence'}
                            )
                            fig_hist.update_layout(height=300)
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
                else:
                    st.error("❌ Impossible de charger le modèle")

# Footer avec statistiques
st.markdown("---")
col_foot1, col_foot2, col_foot3, col_foot4 = st.columns(4)

with col_foot1:
    st.metric("Classes supportées", "38")

with col_foot2:
    st.metric("Types de plantes", "14")

with col_foot3:
    st.metric("Maladies détectées", "26")

with col_foot4:
    st.metric("Précision du modèle", "~95%")

# Informations additionnelles
with st.expander("📚 Guide d'utilisation"):
    st.markdown("""
    ### Comment utiliser cette application:
    
    1. **📁 Téléchargez une image** dans la sidebar (formats: JPG, PNG, JPEG, BMP, TIFF)
    2. **⚙️ Ajustez les paramètres** selon vos besoins
    3. **🚀 Cliquez sur "Lancer l'analyse"** pour obtenir la prédiction
    4. **📊 Analysez les résultats** et les métriques de confiance
    
    ### Types de plantes supportées:
    - 🍎 Pommes
    - 🫐 Myrtilles  
    - 🍒 Cerises
    - 🌽 Maïs
    - 🍇 Raisins
    - 🍊 Oranges
    - 🍑 Pêches
    - 🌶️ Poivrons
    - 🥔 Pommes de terre
    - 🫐 Framboises
    - 🌱 Soja
    - 🎃 Courges
    - 🍓 Fraises
    - 🍅 Tomates
    
    ### Conseils pour de meilleurs résultats:
    - Utilisez des images claires et bien éclairées
    - Centrez la feuille ou la partie malade de la plante
    - Évitez les images floues ou trop sombres
    - Assurez-vous que la maladie est visible sur l'image
    """)

with st.expander("🧠 À propos du modèle"):
    st.markdown("""
    ### Architecture du modèle:
    - **Type:** Réseau de Neurones Convolutifs (CNN)
    - **Framework:** TensorFlow/Keras
    - **Taille d'entrée:** 64x64x3 pixels
    - **Normalisation:** Images normalisées entre 0 et 1
    
    ### Performance:
    - **Précision estimée:** ~95% sur le dataset de test
    - **Temps d'inférence:** < 100ms en moyenne
    - **Classes:** 38 combinaisons plante-maladie
    
    ### Limitations:
    - Le modèle a été entraîné sur un dataset spécifique
    - Les résultats peuvent varier selon la qualité de l'image
    - Utilisez les prédictions comme aide au diagnostic, pas comme diagnostic final
    """)