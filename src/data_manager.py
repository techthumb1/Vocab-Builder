import nltk
import streamlit as st

@st.cache_resource
def initialize_nltk_data():
    """
    Ensures NLTK data (WordNet, POS tagger, tokenizer) is installed
    and loaded in a cached manner so it's not repeated on each run.
    """
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("punkt")
    return True
