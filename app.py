import streamlit as st
import pandas as pd
import joblib  

# Charger les modèles pré-entraînés

random_forest_pipeline = joblib.load('random_forest_pipeline_laptop_price_predictor_model.pkl')

# Interface Streamlit
st.title('Comparaison des modèles de prédiction du prix des laptops')

# Formulaire avec des indications claires
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Sélectionnez la marque', options=['Sélectionner...', 'Dell', 'HP', 'Apple', 'Asus'], index=0)
    type_name = st.selectbox('Sélectionnez le type', options=['Sélectionner...', 'Laptop'], index=0)
    cpu = st.selectbox('Sélectionnez le processeur', options=['Sélectionner...', 'Intel', 'AMD'], index=0)
    gpu = st.selectbox('Sélectionnez la carte graphique', options=['Sélectionner...', 'NVIDIA', 'AMD', 'Intel'], index=0)
    opsys = st.selectbox('Sélectionnez le système d\'exploitation', options=['Sélectionner...', 'Windows', 'macOS', 'Ubuntu'], index=0)

with col2:
    inches = st.number_input('Entrez la taille de l\'écran (en pouces)', min_value=10.0, max_value=20.0, step=0.1)
    ram = st.selectbox('Sélectionnez la mémoire RAM (Go)', options=['Sélectionner...', 4, 8, 16, 32], index=0)
    weight = st.number_input('Entrez le poids (en kg)', min_value=0.5, max_value=3.0, step=0.1)
    width = st.number_input('Entrez la largeur (en cm)', min_value=0.0, max_value=40.0, step=0.1)
    height = st.number_input('Entrez la hauteur (en cm)', min_value=0.0, max_value=40.0, step=0.1)

# Bouton pour valider la prédiction
if st.button('Valider'):
    if company != 'Sélectionner...' and type_name != 'Sélectionner...' and cpu != 'Sélectionner...' and gpu != 'Sélectionner...' and opsys != 'Sélectionner...':
        # Préparer les données d'entrée
        input_data = pd.DataFrame({
            'Company': [company],
            'TypeName': [type_name],
            'Inches': [inches],
            'Ram': [ram],
            'Weight': [weight],
            'Width': [width],
            'Height': [height],
            'Cpu': [cpu],
            'Gpu': [gpu],
            'OpSys': [opsys]
        })

        # Prédictions avec les deux modèles
        rf_prediction = random_forest_pipeline.predict(input_data)

        # Affichage des résultats
        st.subheader('Résultats de prédiction du modèle')

        # Afficher les prédictions des deux modèles
        st.write(f"**Prix prédit par le modèle Random Forest** : {rf_prediction[0]:.2f} ")

    else:
        st.warning("Veuillez remplir tous les champs correctement.")

# Bouton pour réinitialiser les champs
if st.button('Initialiser'):
    st.stop()
