import streamlit as st
import joblib

# Charger le modèle et le vectoriseur
model = joblib.load("modele_logistic_regression.pkl")
vectorizer = joblib.load("vectoriseur_tfidf.pkl")

# Titre de l'application
st.title("📊 Analyse de Sentiment - Avis Allociné")
st.write("Entrez un avis en français et obtenez sa polarité (positif ou négatif).")

# Zone de saisie
avis = st.text_area("✍️ Entrez votre avis ici :", "")

if st.button("Prédire la polarité"):
    if avis.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        # Transformer l'avis en vecteur TF-IDF
        X_new = vectorizer.transform([avis])
        # Prédiction
        prediction = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0]

        # Affichage
        if prediction == 1:
            st.success(f"✅ Avis POSITIF avec une probabilité de {proba[1]*100:.2f}%")
        else:
            st.error(f"❌ Avis NÉGATIF avec une probabilité de {proba[0]*100:.2f}%")
