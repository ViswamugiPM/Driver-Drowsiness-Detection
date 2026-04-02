
import streamlit as st

st.title("🔍 Model Reasoning")

st.markdown("""
## 🧠 How the System Works

### Step 1 — Text Input
- Title + Description combined

### Step 2 — Transformer Encoding
- DistilBERT converts text into embeddings

### Step 3 — Proxy Features
- Emotion
- Urgency
- Readability
- Lexical richness

### Step 4 — Scoring
- Weighted combination of features

---

## 🎯 Why This Works

Popularity is not directly observable.

So we approximate it using:

- Linguistic signals
- Emotional signals
- Structural patterns

---

## ⚡ Key Insight

👉 This is a **weakly supervised system**  
(no real labels used)

---

## 🚀 Real-World Use

- News ranking systems
- Social media feeds
- Content recommendation engines
""")

st.success("This system simulates real AI editorial decision systems")