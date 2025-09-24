import pandas as pd
import numpy as np
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import streamlit as st

st.set_page_config(page_title="Fake vs Real News Detection", layout="wide")

# Load data
data_path = "processed_data.csv"
df = pd.read_csv(data_path)
st.success("Data loaded successfully")

# Load model
MODEL_PATH = "models/best_rf_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("Model not found")
    st.stop()

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Sidebar navigation
choice = st.sidebar.radio("Choose one", ("Dataset Overview", "Visualization", "Prediction", "Evaluation"))

# Clean text and dates
df = df.dropna(subset=["clean_text"])
df["date"] = pd.to_datetime(df["date"], errors="coerce")

st.title("Fake vs Real News Detection")

# Dataset Overview
if choice == "Dataset Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("Column Info")
    dtype_df = df.dtypes.reset_index()
    dtype_df.columns = ['Column', 'Data Type']
    st.dataframe(dtype_df)

    st.subheader("Missing Values")
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ['Column', 'Missing Values']
    st.dataframe(missing_df)

    st.subheader("Sample Titles")
    st.write(df['title'].dropna().sample(5).tolist())
    
# Visualization
elif choice == "Visualization":
    
    label_counts = df["label"].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    label_series = pd.Series(
        [label_counts.get(0, 0), label_counts.get(1, 0)],
        index=["Fake", "Real"]
    )

    label_series.plot(kind='bar', ax=axes[0], color=["#891d1d", "#024604"])
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("Label")
    axes[0].set_title("Fake vs Real")

    subject_counts = df["subject"].value_counts()
    subject_counts.plot(kind="bar", ax=axes[1], color="skyblue")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Subject")
    axes[1].set_title("Subject Frequency")
    axes[1].tick_params(axis='x', rotation=45)

    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Word Cloud of Cleaned Text")
    text = " ".join(df['clean_text'].dropna().astype(str).to_list())

    wc = WordCloud(
        background_color='white',
        height=500,
        width=800,
        stopwords=STOPWORDS
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Prediction
elif choice == "Prediction":
    st.subheader("Predict News Authenticity")

    TFIDF_PATH = "models/tfidf_vectorizer.pkl"
    if not os.path.exists(TFIDF_PATH):
        st.error("TF-IDF vectorizer not found")
        st.stop()

    with open(TFIDF_PATH, 'rb') as f:
        vectorizer = pickle.load(f)

    if df is not None:
        row_index = st.number_input("Select Row Index to Predict", min_value=0, max_value=len(df)-1, value=0)
        input_row = df.iloc[row_index]
        input_text = input_row["clean_text"]
        title = input_row["title"]

        st.markdown(f"**Title:** {title}")

        if st.button("Predict Fake or Real"):
            try:
                input_vector = vectorizer.transform([input_text])
                prediction = model.predict(input_vector)[0]
                label = "Real News" if prediction == 1 else "Fake News"
                st.success(f"Prediction: {label}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.info("Please load data and try again")

# Evaluation
else:
    st.subheader("Model Evaluation on Holdout Test Set")

    TFIDF_PATH = "models/tfidf_vectorizer.pkl"
    if not os.path.exists(TFIDF_PATH):
        st.error("TF-IDF vectorizer not found")
        st.stop()

    with open(TFIDF_PATH, 'rb') as f:
        vectorizer = pickle.load(f)

    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_vec = vectorizer.transform(X_test)

    y_pred = model.predict(X_test_vec)
    y_true = y_test

    acc = accuracy_score(y_true, y_pred)
    st.metric(label="Accuracy", value=f"{acc * 100:.2f}%")

    report = classification_report(y_true, y_pred)
    st.subheader("Classification Report")
    st.text(report)

    cm = confusion_matrix(y_true, y_pred)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize= (14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    ax.set_ylabel("Actual")
    st.pyplot(fig)
