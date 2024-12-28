import os
import nltk
import streamlit as st

@st.cache_resource
def initialize_nltk_data():
    nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))
    """
    Ensures NLTK data (WordNet, POS tagger, tokenizer) is installed
    and loaded in a cached manner so it's not repeated on each run.
    """
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("punkt", quiet=True)
    return True
