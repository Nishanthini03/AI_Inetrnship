**Airline Sentiment Analysis Using NLP & Machine Learning**

📌 **Project Overview**

This project aims to analyze airline-related tweets and classify them into Positive, Neutral, or Negative sentiments using Natural Language Processing (NLP) and Machine Learning techniques. Additionally, an interactive Streamlit web application has been developed for real-time sentiment analysis.

✨ **Key Features**

Data Preprocessing: Cleaning text by removing URLs, mentions, hashtags, and stopwords.

Exploratory Data Analysis (EDA): Visualizing sentiment distribution, word cloud, and tweet length analysis.

Feature Engineering: Implementing TF-IDF vectorization to convert text into numerical representations.

Model Training & Evaluation:

Naive Bayes Classifier

Random Forest Classifier

Deployment: A user-friendly Streamlit web app for real-time sentiment classification.

🔧 **Technologies Used**

Programming Language: Python

Libraries:

NLP: NLTK, spaCy

Data Analysis & Visualization: Pandas, NumPy, Matplotlib, Seaborn

Machine Learning: Scikit-learn

Deployment: Streamlit

📂 **Dataset Information**

The dataset consists of airline-related tweets with sentiment labels (Positive, Neutral, and Negative). It includes attributes such as:

text: The tweet content

airline_sentiment: The sentiment label

airline: The airline associated with the tweet

negativereason: The reason for negative sentiment (if applicable)

tweet_created: The timestamp of the tweet

🚀 **Installation & Execution**

1️⃣ Install Dependencies

pip install pandas numpy nltk scikit-learn streamlit matplotlib seaborn wordcloud

2️⃣ Download NLTK Resources (Run in Python)

import nltk
nltk.download('stopwords')
nltk.download('punkt')

3️⃣ Run the Streamlit Web Application

streamlit run app.py
