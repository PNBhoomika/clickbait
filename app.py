import pandas as pd
import pickle
import streamlit as st

st.title("Clickbait Detector")
st.success("Deployment is working!")
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("dataset.csv")

# Remove missing values
df = df.dropna()

# Features and Labels
X = df["headline"]
y = df["clickbait"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model
model = PassiveAggressiveClassifier(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# Accuracy
predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", round(accuracy*100,2), "%")

# Save Model
pickle.dump(
    model,
    open("model.pkl","wb")
)

pickle.dump(
    vectorizer,
    open("vectorizer.pkl","wb")
)

print("Model Saved Successfully")
