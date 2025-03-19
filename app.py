import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'@\w+', '', text)  # Remove mentions (@user)
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit App
st.title("✈️ Airline Sentiment Analyzer")
st.write("Enter a tweet related to airlines and get instant sentiment feedback!")

# User Input
user_input = st.text_area("Enter a tweet:", "")

if st.button("Analyze Sentiment"):
    if user_input:
        # Clean and transform input text
        cleaned_input = clean_text(user_input)
        input_features = vectorizer.transform([cleaned_input])

        # Predict sentiment
        prediction = model.predict(input_features)[0]

        # Convert numeric label to sentiment
        sentiment_labels = {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😊"}
        st.subheader(f"Sentiment: {sentiment_labels[prediction]}")
    else:
        st.warning("Please enter a tweet before analyzing.")
