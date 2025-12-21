# ======================================================
# 🌪️ TextVortex — Advanced Real-Time NLP Intelligence
# ======================================================

# ------------------------------------------------------
# ✅ BULLETPROOF NLTK SETUP (Python 3.13 + Streamlit)
# ------------------------------------------------------
import nltk
import streamlit as st

@st.cache_resource
def setup_nltk():
    resources = [
        "tokenizers/punkt",
        "tokenizers/punkt_tab/english",
        "corpora/stopwords",
        "corpora/wordnet",
        "corpora/omw-1.4",
        "taggers/averaged_perceptron_tagger"
    ]

    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1])

setup_nltk()


# ------------------------------------------------------
# NLP IMPORTS (AFTER DOWNLOAD)
# ------------------------------------------------------
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import FreqDist, pos_tag
from nltk.util import ngrams

# ------------------------------------------------------
# OTHER IMPORTS
# ------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
import re
import base64

# ------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="TextVortex 🌪️",
    page_icon="🌪️",
    layout="wide"
)

st.title("🌪️ TextVortex — Advanced Real-Time NLP Intelligence")

# ------------------------------------------------------
# GLOBAL CONSTANTS
# ------------------------------------------------------
STOPWORDS = set(stopwords.words("english"))

# ------------------------------------------------------
# SAFE CONTRACTION EXPANDER (NO EXTERNAL LIBRARY)
# ------------------------------------------------------
def expand_contractions(text):
    contractions_map = {
        "can't": "cannot", "won't": "will not", "don't": "do not",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not",
        "weren't": "were not", "hasn't": "has not", "haven't": "have not",
        "hadn't": "had not", "doesn't": "does not", "didn't": "did not",
        "i'm": "i am", "you're": "you are", "they're": "they are",
        "we're": "we are", "it's": "it is", "that's": "that is"
    }
    for k, v in contractions_map.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
    return text

# ------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------
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
        "🧮 Text Normalization",
        "🔢 N-Grams",
        "🔑 Keyword Extraction",
        "🔄 Text Similarity",
        "📈 Text Complexity",
        "☁️ Word Cloud"
    ]
)

st.sidebar.subheader("📝 Input Text")
text_input = st.sidebar.text_area("Enter text", height=200)

if st.sidebar.button("Load Text"):
    st.session_state.text = text_input

text = st.session_state.get("text", "")

# ------------------------------------------------------
# HOME
# ------------------------------------------------------
if page == "🏠 Home":
    st.markdown("""
### 🌪️ What is TextVortex?

**TextVortex** is a unified, real-time Natural Language Processing (NLP) intelligence platform  
designed for **research, education, and explainable text analytics**.

✔ Cloud-safe  
✔ Conference-ready  
✔ Modular NLP pipeline  
✔ Future-proof architecture  
""")

# ------------------------------------------------------
# TEXT STATISTICS
# ------------------------------------------------------
elif page == "📊 Text Statistics":
    if text:
        st.metric("Characters", len(text))
        st.metric("Words", len(text.split()))
        st.metric("Sentences", len(sent_tokenize(text)))
    else:
        st.warning("Please load text")

# ------------------------------------------------------
# TOKENIZATION
# ------------------------------------------------------
elif page == "🔠 Tokenization":
    if text:
        st.write(word_tokenize(text))
    else:
        st.warning("Please load text")

# ------------------------------------------------------
# STOPWORD REMOVAL
# ------------------------------------------------------
elif page == "🛑 Stopword Removal":
    if text:
        tokens = word_tokenize(text)
        filtered = [w for w in tokens if w.lower() not in STOPWORDS]
        st.write(filtered)
    else:
        st.warning("Please load text")

# ------------------------------------------------------
# POS TAGGING
# ------------------------------------------------------
elif page == "🏷️ POS Tagging":
    if text:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        st.dataframe(pd.DataFrame(tagged, columns=["Word", "POS"]))
    else:
        st.warning("Please load text")

# ------------------------------------------------------
# STEMMING
# ------------------------------------------------------
elif page == "🌱 Stemming":
    if text:
        stemmer = PorterStemmer()
        tokens = word_tokenize(text)
        st.write([stemmer.stem(w) for w in tokens])
    else:
        st.warning("Please load text")

# ------------------------------------------------------
# LEMMATIZATION
# ------------------------------------------------------
elif page == "🌿 Lemmatization":
    if text:
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        st.write([lemmatizer.lemmatize(w) for w in tokens])
    else:
        st.warning("Please load text")

# ------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------
elif page == "🧮 Text Normalization":
    if text:
        t = text.lower()
        t = expand_contractions(t)
        t = re.sub(r"[^\w\s]", "", t)
        tokens = [w for w in word_tokenize(t) if w not in STOPWORDS]
        st.write(" ".join(tokens))
    else:
        st.warning("Please load text")

# ------------------------------------------------------
# N-GRAMS
# ------------------------------------------------------
elif page == "🔢 N-Grams":
    if text:
        n = st.slider("Select N", 1, 3, 2)
        tokens = word_tokenize(text)
        st.write(list(ngrams(tokens, n)))
    else:
        st.warning("Please load text")

# ------------------------------------------------------
# KEYWORD EXTRACTION
# ------------------------------------------------------
elif page == "🔑 Keyword Extraction":
    if text:
        tokens = [w.lower() for w in word_tokenize(text) if w.isalnum()]
        freq = FreqDist(tokens)
        df = pd.DataFrame(freq.items(), columns=["Keyword", "Frequency"])
        st.dataframe(df.sort_values("Frequency", ascending=False).head(20))
    else:
        st.warning("Please load text")

# ------------------------------------------------------
# TEXT SIMILARITY
# ------------------------------------------------------
elif page == "🔄 Text Similarity":
    t1 = st.text_area("Text 1")
    t2 = st.text_area("Text 2")
    if st.button("Compare"):
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([t1, t2])
        score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        st.metric("Cosine Similarity", round(score, 3))

# ------------------------------------------------------
# TEXT COMPLEXITY
# ------------------------------------------------------
elif page == "📈 Text Complexity":
    if text:
        st.metric("Flesch Reading Ease", round(textstat.flesch_reading_ease(text), 2))
        st.metric("Gunning Fog Index", round(textstat.gunning_fog(text), 2))
        st.metric("Grade Level", textstat.text_standard(text))
    else:
        st.warning("Please load text")

# ------------------------------------------------------
# WORD CLOUD
# ------------------------------------------------------
elif page == "☁️ Word Cloud":
    if text:
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.warning("Please load text")

