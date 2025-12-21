# =========================
# 🔹 CORE IMPORTS
# =========================
import streamlit as st
import nltk

# =========================
# 🔹 SAFE NLTK DOWNLOADER (MUST BE FIRST)
# =========================
@st.cache_resource
def download_nltk_resources():
    resources = [
        "punkt",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger"
    ]
    for r in resources:
        try:
            nltk.data.find(r)
        except LookupError:
            nltk.download(r)

download_nltk_resources()

# =========================
# 🔹 NLP IMPORTS (AFTER DOWNLOAD)
# =========================
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import FreqDist, pos_tag
from nltk.util import ngrams

# =========================
# 🔹 DATA & VISUAL IMPORTS
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
import re
import contractions
import base64

# =========================
# 🔹 GLOBAL CONSTANTS
# =========================
STOPWORDS = set(stopwords.words("english"))

# =========================
# 🔹 STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="TextVortex 🌪️",
    page_icon="🌪️",
    layout="wide"
)

st.title("🌪️ TextVortex — Advanced Real-Time NLP Intelligence")

# =========================
# 🔹 SIDEBAR
# =========================
page = st.sidebar.radio(
    "🌐 Select Feature",
    [
        "🏠 Home",
        "📊 Text Statistics",
        "🔠 Tokenization",
        "🛑 Stopword Removal",
        "🏷️ POS Tagging",
        "🌱 Stemming",
        "🌿 Lemmatization",
        "🧮 Normalization",
        "🔢 N-Grams",
        "🔑 Keyword Extraction",
        "🔄 Text Similarity",
        "📈 Text Complexity",
        "☁️ Word Cloud"
    ]
)

# =========================
# 🔹 INPUT HANDLER
# =========================
if "text_data" not in st.session_state:
    st.session_state.text_data = ""

st.sidebar.subheader("📝 Input Text")
user_text = st.sidebar.text_area("Enter text", height=200)

if st.sidebar.button("✅ Load Text"):
    st.session_state.text_data = user_text

text = st.session_state.text_data

# =========================
# 🔹 HOME
# =========================
if page == "🏠 Home":
    st.markdown("""
    ### 🌪️ What is TextVortex?
    **TextVortex** is a unified, real-time NLP analysis platform designed for  
    **research, education, and intelligent text exploration**.

    ✔ Cloud-safe  
    ✔ Explainable NLP  
    ✔ Conference-ready  
    ✔ Modular & future-proof  
    """)

# =========================
# 🔹 TEXT STATS
# =========================
elif page == "📊 Text Statistics":
    if text:
        st.metric("Characters", len(text))
        st.metric("Words", len(text.split()))
        st.metric("Sentences", len(sent_tokenize(text)))
    else:
        st.warning("Enter text first")

# =========================
# 🔹 TOKENIZATION
# =========================
elif page == "🔠 Tokenization":
    if text:
        st.write(word_tokenize(text))
    else:
        st.warning("Enter text first")

# =========================
# 🔹 STOPWORDS
# =========================
elif page == "🛑 Stopword Removal":
    if text:
        tokens = word_tokenize(text)
        filtered = [w for w in tokens if w.lower() not in STOPWORDS]
        st.write(filtered)
    else:
        st.warning("Enter text first")

# =========================
# 🔹 POS TAGGING
# =========================
elif page == "🏷️ POS Tagging":
    if text:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        df = pd.DataFrame(tagged, columns=["Word", "POS"])
        st.dataframe(df)
    else:
        st.warning("Enter text first")

# =========================
# 🔹 STEMMING
# =========================
elif page == "🌱 Stemming":
    if text:
        stemmer = PorterStemmer()
        tokens = word_tokenize(text)
        st.write([stemmer.stem(w) for w in tokens])
    else:
        st.warning("Enter text first")

# =========================
# 🔹 LEMMATIZATION
# =========================
elif page == "🌿 Lemmatization":
    if text:
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        st.write([lemmatizer.lemmatize(w) for w in tokens])
    else:
        st.warning("Enter text first")

# =========================
# 🔹 NORMALIZATION
# =========================
elif page == "🧮 Normalization":
    if text:
        t = text.lower()
        t = contractions.fix(t)
        t = re.sub(r"[^\w\s]", "", t)
        tokens = [w for w in word_tokenize(t) if w not in STOPWORDS]
        st.write(" ".join(tokens))
    else:
        st.warning("Enter text first")

# =========================
# 🔹 N-GRAMS
# =========================
elif page == "🔢 N-Grams":
    if text:
        n = st.slider("N", 1, 3, 2)
        tokens = word_tokenize(text)
        st.write(list(ngrams(tokens, n)))
    else:
        st.warning("Enter text first")

# =========================
# 🔹 KEYWORDS
# =========================
elif page == "🔑 Keyword Extraction":
    if text:
        tokens = [w.lower() for w in word_tokenize(text) if w.isalnum()]
        freq = FreqDist(tokens)
        df = pd.DataFrame(freq.items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)
        st.dataframe(df.head(20))
    else:
        st.warning("Enter text first")

# =========================
# 🔹 SIMILARITY
# =========================
elif page == "🔄 Text Similarity":
    t1 = st.text_area("Text 1")
    t2 = st.text_area("Text 2")
    if st.button("Compare"):
        vec = TfidfVectorizer()
        tfidf = vec.fit_transform([t1, t2])
        score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        st.metric("Similarity Score", round(score, 3))

# =========================
# 🔹 COMPLEXITY
# =========================
elif page == "📈 Text Complexity":
    if text:
        st.metric("Flesch Reading Ease", textstat.flesch_reading_ease(text))
        st.metric("Gunning Fog Index", textstat.gunning_fog(text))
        st.metric("Grade Level", textstat.text_standard(text))
    else:
        st.warning("Enter text first")

# =========================
# 🔹 WORD CLOUD
# =========================
elif page == "☁️ Word Cloud":
    if text:
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.warning("Enter text first")
