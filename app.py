import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from src.data_manager import initialize_nltk_data
from src.dict_enrichment import (
    fetch_dictionary_data,
    parse_pronunciation,
    parse_etymology,
    parse_meanings
)
from src.synonyms import synonyms_from_wordnet, contextual_synonyms
from src.embeddings import TransformerSynonymFinder
from src.llm_integration import llm_writing_advice
from src.feedback import record_like, get_likes

initialize_nltk_data()  # 


# Main function for the Streamlit app
def main():
    st.title("WordSage")
    st.caption("A multi-faceted word analysis tool")
    st.write("""
    This app offers:
    1. Dictionary enrichment with pronunciation, etymology, usage.
    2. WordNet synonyms (basic & contextual).
    3. Optional advanced synonyms ranking with transformer embeddings.
    4. LLM-based writing advice (synonyms, usage contexts, structure tips).
    """)

    # Initialize NLTK data
    initialize_nltk_data()

    # Initialize Synonym Finder
    finder = TransformerSynonymFinder()

    # Input fields
    user_word = st.text_input(
        label="Word",
        placeholder="Enter a word (e.g. 'sesquipedalian')"
    )

    user_sentence = st.text_input(
        label="Contextual Sentence",
        placeholder="Enter a sentence for context"
    )

    # Analyze button
    if st.button("Analyze Word"):
        if not user_word.strip():
            st.warning("Please enter a valid word.")
            return

        st.subheader("Dictionary Enrichment")
        dict_data = fetch_dictionary_data(user_word.lower())
        if dict_data:
            pron = parse_pronunciation(dict_data)
            if pron:
                st.write(f"**Pronunciation**: {pron}")

            origin = parse_etymology(dict_data)
            if origin:
                st.write(f"**Etymology**: {origin}")

            meanings = parse_meanings(dict_data)
            if meanings:
                st.write("**Detailed Meanings**:")
                for pos, entries in meanings.items():
                    st.markdown(f"**{pos}**")
                    for i, e in enumerate(entries, start=1):
                        st.write(f"- Definition {i}: {e['definition']}")
                        if e["example"]:
                            st.write(f"  Example: _{e['example']}_")
                        if e["synonyms"]:
                            st.write(f"  Synonyms: {e['synonyms']}")
        else:
            st.write("No dictionary data found from the external API.")

        st.subheader("WordNet Synonyms")
        basic_syns = synonyms_from_wordnet(user_word.lower())
        if basic_syns:
            st.write(basic_syns)
            if user_sentence.strip():
                ranked_syns = finder.rank_synonyms(basic_syns, context=user_sentence, top_n=5)
                st.subheader("Advanced Synonyms (Embedding-based)")
                if ranked_syns:
                    for syn in ranked_syns:
                        like_count = get_likes(user_word, syn)
                        st.write(f"- {syn} (likes: {like_count})")
                        if st.button(f"Like '{syn}'", key=f"like_{syn}"):
                            record_like(user_word, syn)
                            st.experimental_rerun()
        else:
            st.write("No basic synonyms found.")

        if user_sentence.strip():
            st.subheader("Contextual Synonyms (POS-based)")
            c_syns = contextual_synonyms(user_sentence, user_word.lower())
            if c_syns:
                st.write(c_syns)
            else:
                st.write("No contextual synonyms found for that sentence.")

    # Generate Enhancements button
    if st.button("Generate Enhancements"):
        if not user_word.strip():
            st.warning("Please enter a word first.")
            return
        st.subheader("LLM-Based Writing Advice")
        with st.spinner("Generating advice..."):
            try:
                advice_text = llm_writing_advice(user_word, user_sentence)
                st.write(advice_text)
            except Exception as e:
                st.error(f"Error generating advice: {e}")

if __name__ == "__main__":
    main()
