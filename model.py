import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from transformers import pipeline

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

try:
    svm_model = joblib.load("svm_sentiment_model.pkl")
    log_reg_model = joblib.load("log_reg_sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except:
    svm_model, log_reg_model, vectorizer = None, None, None

sentiment_analysis = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", framework="pt")

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return " ".join([para.text for para in doc.paragraphs])
    elif uploaded_file.type in ["text/plain", "text/csv", "application/vnd.ms-excel"]:
        return uploaded_file.getvalue().decode("utf-8")
    return ""

def analyze_sentiment_finbert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
    return {"positive": scores[0], "neutral": scores[1], "negative": scores[2]}

def analyze_sentiment_ml(text, model_type):
    if not vectorizer:
        return "ML models are not loaded. Train and upload them first."
    text_vectorized = vectorizer.transform([text])
    if model_type == "SVM":
        prediction = svm_model.predict(text_vectorized)[0]
    else:
        prediction = log_reg_model.predict(text_vectorized)[0]
    return "Positive" if prediction == 1 else "Negative"

def analyze_sentiment_vader(text):
    scores = sia.polarity_scores(text)
    return scores

def analyze_sentiment_llm(text):
    result = sentiment_analysis(text)
    return result[0]

def analyze_sentiment_manual_dict(text):
    positive_words = ["good", "great", "excellent", "positive", "strong", "growth", "increase", "profit", "success", "stable"]
    negative_words = ["bad", "poor", "negative", "weak", "decline", "loss", "risk", "volatile", "uncertainty", "failure"]

    text_lower = text.lower()
    positive_count = sum(text_lower.count(word) for word in positive_words)
    negative_count = sum(text_lower.count(word) for word in negative_words)

    if positive_count > negative_count:
        return {"sentiment": "Positive", "positive_count": positive_count, "negative_count": negative_count}
    elif negative_count > positive_count:
        return {"sentiment": "Negative", "positive_count": positive_count, "negative_count": negative_count}
    else:
        return {"sentiment": "Neutral", "positive_count": positive_count, "negative_count": negative_count}

def classify_risk(sentiment_score):
    if isinstance(sentiment_score, dict):
        compound = sentiment_score.get('compound', 0)
    elif isinstance(sentiment_score, str):
        if sentiment_score == 'Positive':
            compound = 1
        else:
            compound = -1
    else:
        compound = 0

    if compound < -0.5:
        return "High Risk"
    elif -0.5 <= compound <= -0.2:
        return "Moderate Risk"
    elif -0.2 <= compound <= 0.2:
        return "Neutral"
    else:
        return "Stable"

def filter_text_by_keyword(text, keyword):
    filtered_sentences = [sentence for sentence in text.split('. ') if keyword.lower() in sentence.lower()]
    return "\n".join(filtered_sentences) if filtered_sentences else "No relevant sentences found."

st.set_page_config(page_title="Financial Sentiment Analysis", layout="wide")
st.title("📊 Financial Sentiment Analysis Dashboard")

uploaded_file = st.file_uploader("📂 Upload a financial document (PDF, DOCX, TXT, CSV, Excel)", type=["pdf", "docx", "txt", "csv", "xls", "xlsx"])

if uploaded_file:
    text = extract_text_from_file(uploaded_file)
    st.subheader("📜 Extracted Text Preview")
    st.text_area("", text[:1000], height=150)

    keyword = st.text_input("🔍 Enter a keyword to filter the text:")
    if keyword:
        filtered_text = filter_text_by_keyword(text, keyword)
        st.subheader("🔎 Filtered Text")
        st.text_area("", filtered_text, height=150)

    model_choice = st.selectbox("Select Sentiment Analysis Model:", ["FinBERT", "SVM", "Logistic Regression", "VADER", "Manual Dictionary", "LLM"])

    if st.button("🚀 Analyze Sentiment"):
        st.subheader(f"📈 {model_choice} Sentiment Analysis Results")

        if model_choice == "FinBERT":
            sentiment_scores = analyze_sentiment_finbert(text)
        elif model_choice in ["SVM", "Logistic Regression"]:
            sentiment_scores = {"sentiment": analyze_sentiment_ml(text, model_choice)}
        elif model_choice == "VADER":
            sentiment_scores = analyze_sentiment_vader(text)
        elif model_choice == "Manual Dictionary":
            sentiment_scores = analyze_sentiment_manual_dict(text)
        elif model_choice == "LLM":
            sentiment_scores = analyze_sentiment_llm(text)

        st.json(sentiment_scores)

        risk_level = classify_risk(sentiment_scores)
        st.subheader("⚠️ Risk Level")
        st.markdown(f"**Risk Classification:** `{risk_level}`")

        st.subheader("📊 Sentiment Score Distribution")

        if isinstance(sentiment_scores, dict):
            fig, ax = plt.subplots()
            ax.bar(sentiment_scores.keys(), sentiment_scores.values(), color=['green', 'gray', 'red', 'blue'])
            ax.set_title("Sentiment Score Distribution")
            st.pyplot(fig)
        elif isinstance(sentiment_scores, str):
            st.write(f"Sentiment: {sentiment_scores}")
        else:
            st.write("Sentiment analysis result cannot be displayed in chart format.")
