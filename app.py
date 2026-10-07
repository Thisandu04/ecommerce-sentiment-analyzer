import streamlit as st
import joblib
import sys
sys.path.append(".")
from src.preprocessing import clean_text

# Cache so the model only loads once, not on every interaction
@st.cache_resource
def load_pipeline():
    vectorizer = joblib.load("models/vectorizer.pkl")
    model = joblib.load("models/classifier.pkl")
    return vectorizer, model

vectorizer, model = load_pipeline()

st.title("E-commerce Review Sentiment Analyzer")
st.write("Paste a product review below to see its predicted sentiment.")

review_text = st.text_area("Review text", height=150)

if st.button("Predict Sentiment"):
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(review_text)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities)

        color = {"Positive": "green", "Neutral": "orange", "Negative": "red"}[prediction]
        st.markdown(f"### Prediction: :{color}[{prediction}]")
        st.write(f"Confidence: {confidence:.1%}")

        with st.expander("See full probability breakdown"):
            for label, prob in zip(model.classes_, probabilities):
                st.write(f"{label}: {prob:.1%}")