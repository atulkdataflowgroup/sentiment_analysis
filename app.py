import streamlit as st
import nltk
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Load RoBERTa model for sentiment analysis
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def get_sentiment_label(roberta_result, message):
    roberta_label = roberta_result[0]['label']
    roberta_score = roberta_result[0]['score']
    vader_score = sia.polarity_scores(message)["compound"]
    
    if roberta_label == "positive" and roberta_score < 0.8:
        roberta_label = "neutral"
    
    if roberta_score > 0.8:
        return roberta_label.capitalize()
    
    if vader_score >= 0.4:
        return "Positive"
    elif vader_score <= -0.4:
        return "Negative"
    else:
        return "Neutral"

def analyze_user_sentiment_from_file(file):
    chat_transcript = file.read().decode("utf-8").splitlines()
    cleaned_transcript = []
    skip_lines = ["Conversation Log", "Assistant: Hello,👋 I'm Vincent, your DataFlow virtual assistant and I'll do my best to get you the answers you need. How can I assist you today?"]
    
    for line in chat_transcript:
        stripped_line = line.strip()
        if stripped_line and stripped_line not in skip_lines:
            cleaned_transcript.append(stripped_line)
    
    user_messages = [msg for msg in cleaned_transcript if msg.startswith("User:")]
    
    positive_count = negative_count = neutral_count = 0
    user_message_sentiments = []
    
    for message in user_messages:
        roberta_result = sentiment_pipeline(message)
        label = get_sentiment_label(roberta_result, message)
        user_message_sentiments.append((message, label))
        
        if label == "Positive":
            positive_count += 1
        elif label == "Negative":
            negative_count += 1
        else:
            neutral_count += 1
    
    total_messages = len(user_messages)
    sentiment_summary = {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "positive_percentage": (positive_count / total_messages) * 100 if total_messages > 0 else 0,
        "negative_percentage": (negative_count / total_messages) * 100 if total_messages > 0 else 0,
        "neutral_percentage": (neutral_count / total_messages) * 100 if total_messages > 0 else 0,
        "total_messages": total_messages
    }
    
    if positive_count > negative_count and positive_count > neutral_count:
        final_sentiment = "Overall User Sentiment: Positive 😊"
        sentiment_class = "positive"
    elif negative_count > positive_count and negative_count > neutral_count:
        final_sentiment = "Overall User Sentiment: Negative 😞"
        sentiment_class = "negative"
    else:
        final_sentiment = "Overall User Sentiment: Neutral 😐"
        sentiment_class = "neutral"
    
    return sentiment_summary, user_message_sentiments, final_sentiment, sentiment_class

# Streamlit UI Configuration
st.set_page_config(page_title="Sentiment Dashboard", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
        .title {
            font-size: 38px !important;
            font-weight: bold;
            text-align: center;
        }
        .subtitle {
            font-size: 18px !important;
            text-align: center;
            color: grey;
        }
        .sentiment-box {
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            margin-top: 20px;
        }
        .positive { background-color: #D4EDDA; color: #155724; }
        .negative { background-color: #F8D7DA; color: #721C24; }
        .neutral { background-color: #FFF3CD; color: #856404; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<p class="title">📊 Chat Sentiment Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a chatbot conversation file to analyze the overall user sentiment.</p>', unsafe_allow_html=True)

# Sidebar for file upload
with st.sidebar:
    st.header("📂 Upload Chat File")
    uploaded_file = st.file_uploader("Choose a chatbot conversation file (TXT)", type=["txt"])

# Display Results
if uploaded_file:
    with st.spinner("🚀 Analyzing sentiment..."):
        sentiment_results, user_message_sentiments, final_sentiment, sentiment_class = analyze_user_sentiment_from_file(uploaded_file)

    # Sentiment Box with Styling
    st.markdown(f'<div class="sentiment-box {sentiment_class}">{final_sentiment}</div>', unsafe_allow_html=True)
