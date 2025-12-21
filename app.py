import streamlit as st
import re
import base64
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import textstat

# ---------------- CONFIG ---------------- #
st.set_page_config(
    page_title="LexiFlow",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 LexiFlow")
st.caption("Real-Time Linguistic Intelligence Engine")

# ---------------- NLP CORE ---------------- #
STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS]
    return tokens

def realtime_stats(text):
    return {
        "Characters": len(text),
        "Words": len(word_tokenize(text)),
        "Sentences": len(sent_tokenize(text)),
        "Readability": round(textstat.flesch_reading_ease(text), 2)
    }

def topic_modeling(text):
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=1, random_state=42)
    lda.fit(X)
    words = np.array(vectorizer.get_feature_names_out())
    topic = words[np.argsort(lda.components_[0])[-10:]]
    return topic.tolist()

def semantic_similarity(t1, t2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([t1, t2])
    return cosine_similarity(tfidf[0], tfidf[1])[0][0]

def download(text, name):
    b64 = base64.b64encode(text.encode()).decode()
    st.markdown(
        f'<a href="data:file/txt;base64,{b64}" download="{name}">⬇️ Download</a>',
        unsafe_allow_html=True
    )

# ---------------- SIDEBAR ---------------- #
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Live Analyzer",
        "📊 Linguistic Intelligence",
        "🧠 Semantic Analysis",
        "☁️ Concept Visualization"
    ]
)

# ---------------- LIVE ANALYZER ---------------- #
if page == "🏠 Live Analyzer":
    st.subheader("🟢 Real-Time Text Analyzer")

    text = st.text_area("Type or paste text (Live)", height=200)

    if text.strip():
        stats = realtime_stats(text)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Characters", stats["Characters"])
        col2.metric("Words", stats["Words"])
        col3.metric("Sentences", stats["Sentences"])
        col4.metric("Readability", stats["Readability"])

        tokens = normalize(text)
        st.write("🔍 Normalized Tokens:")
        st.write(tokens[:50])

        download(" ".join(tokens), "normalized_text.txt")

# ---------------- LINGUISTIC INTELLIGENCE ---------------- #
elif page == "📊 Linguistic Intelligence":
    st.subheader("📊 Linguistic Intelligence Layer")

    text = st.text_area("Input Text")

    if st.button("Analyze"):
        tokens = normalize(text)
        freq = pd.Series(tokens).value_counts().head(15)

        st.write("🔑 Dominant Linguistic Units")
        st.dataframe(freq)

        st.subheader("📈 Distribution")
        fig, ax = plt.subplots()
        freq.plot(kind="bar", ax=ax)
        st.pyplot(fig)

# ---------------- SEMANTIC ANALYSIS ---------------- #
elif page == "🧠 Semantic Analysis":
    st.subheader("🧠 Semantic Intelligence")

    t1 = st.text_area("Text A")
    t2 = st.text_area("Text B")

    if st.button("Compute Semantic Similarity"):
        score = semantic_similarity(t1, t2)
        st.metric("Similarity Score", f"{score:.3f}")

        st.subheader("📌 Latent Topic (Text A)")
        st.write(topic_modeling(t1))

# ---------------- CONCEPT VISUALIZATION ---------------- #
elif page == "☁️ Concept Visualization":
    st.subheader("☁️ Conceptual Word Cloud")

    text = st.text_area("Text for Visualization")

    if st.button("Generate"):
        tokens = normalize(text)
        wc = WordCloud(width=800, height=400, background_color="black").generate(" ".join(tokens))
        plt.imshow(wc)
        plt.axis("off")
        st.pyplot(plt)
