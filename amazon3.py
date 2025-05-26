import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score
import nltk
from nltk.stem.porter import PorterStemmer
import joblib
from joblib import dump, load

nltk.download('stopwords')

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\HP\OneDrive\Documents\Internship\Project\Amazon_Unlocked_Mobile.csv")
    df = df.dropna(subset=["Reviews", "Rating"])  
    df = df.sample(n=10000, random_state=42)  
    return df

def preprocess_data(df):
    df["Sentiment"] = df["Rating"].apply(lambda x: 0 if x in [1, 2] else (1 if x in [4, 5] else 2))
    
    stop_words = set(nltk.corpus.stopwords.words('english'))
    stemmer = PorterStemmer()
    
    def preprocess_text(text):
        words = text.lower().split()
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return " ".join(words)

    df["Processed_Reviews"] = df["Reviews"].fillna("").apply(preprocess_text)
    return df

def check_negative_phrases(review):
    negative_phrases = ["not satisfied", "worst", "disappointed", "bad", "poor", "horrible", "terrible"]
    review_lower = review.lower()
    for phrase in negative_phrases:
        if phrase in review_lower:
            return 0  
    return None

@st.cache_resource
def train_model(X_train, y_train):
    vectorizer = HashingVectorizer(stop_words='english', n_features=10000)
    clf = Pipeline([("tfidf", vectorizer), ("clf", LinearSVC())])
    clf.fit(X_train, y_train)
    return clf

def main():
    st.title("Amazon Review Sentiment Analysis")

    tab1, tab2 = st.tabs(["Sentiment Analysis", "Review History"])

    if "history" not in st.session_state:
        st.session_state["history"] = []

    df = load_data()
    data_filtered = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        data_filtered["Processed_Reviews"], data_filtered["Sentiment"], test_size=0.2, random_state=42
    )
    clf = train_model(X_train, y_train)

    with tab1:
        rating = st.slider("Select a rating between 1 and 5 stars", 1, 5, 3)
        st.write(f"You selected {rating} stars.")

        user_review = st.text_area("Enter a review for sentiment analysis:")
        
        if user_review:
            rule_based_sentiment = check_negative_phrases(user_review)
            if rule_based_sentiment is not None:
                pred_sentiment = rule_based_sentiment
            else:
                stop_words = set(nltk.corpus.stopwords.words('english'))
                stemmer = PorterStemmer()
                
                def preprocess_text(text):
                    words = text.lower().split()
                    words = [stemmer.stem(word) for word in words if word not in stop_words]
                    return " ".join(words)

                processed_review = preprocess_text(user_review)
                pred_sentiment = clf.predict([processed_review])[0]

            sentiment = 'Negative' if pred_sentiment == 0 else 'Moderate' if pred_sentiment == 2 else 'Positive'
            
            st.session_state["history"].append({
                "Review": user_review,
                "Rating": rating,
                "Sentiment": sentiment
            })

            st.write(f"Review: {user_review}")
            st.write(f"Rating: {rating} stars")
            st.write(f"Predicted Sentiment: {sentiment}")

            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy:.2f}")
			
        elif rating:
            sentiment = 'Negative' if rating in [1, 2] else 'Moderate' if rating == 3 else 'Positive'
            st.write(f"Sentiment: {sentiment} (Rating: {rating})")

    with tab2:
        st.subheader("Review History")
        if st.session_state["history"]:
            st.table(st.session_state["history"])
        else:
            st.write("No reviews analyzed yet.")

if __name__ == "__main__":
    main()