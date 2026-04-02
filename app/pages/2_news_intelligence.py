
import streamlit as st
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from nltk.sentiment import SentimentIntensityAnalyzer

st.title("🧠 News Intelligence")

title = st.text_input("📰 Enter News Title")
description = st.text_area("📄 Enter News Description")

if st.button("🚀 Predict Popularity"):

    with st.spinner("Analyzing article..."):

        text = title + " " + description

        # Load model
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**tokens)

        embedding = outputs.last_hidden_state.mean(dim=1).numpy()

        # Proxy features
        sia = SentimentIntensityAnalyzer()
        emotion = abs(sia.polarity_scores(text)['compound'])

        urgency_words = ["breaking", "urgent", "alert", "shocking"]
        urgency = sum(word in text.lower() for word in urgency_words)

        length = len(text.split())

        # Final score (simple demo logic)
        score = float(np.mean(embedding)) + emotion + urgency * 0.1

    # 🎯 Display Score
    st.subheader("📊 Popularity Score")

    st.progress(min(max(score, 0.0), 1.0))

    st.metric("Score", f"{score:.4f}")

    # 🔍 Explanation Section
    st.subheader("🔍 Model Explanation")

    col1, col2, col3 = st.columns(3)

    col1.metric("Emotion", f"{emotion:.2f}")
    col2.metric("Urgency", urgency)
    col3.metric("Length", length)

    st.info("""
    📌 Higher popularity is influenced by:
    - Emotional intensity
    - Urgency keywords
    - Rich semantic content
    """)

    # 🎯 Interpretation
    if score > 0.7:
        st.success("🔥 High Popularity Potential")
    elif score > 0.4:
        st.warning("⚡ Medium Popularity")
    else:
        st.error("📉 Low Popularity")