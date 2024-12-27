import streamlit as st
from src.data_manager import initialize_nltk_data
from src.definitions import definitions_from_wordnet, examples_from_wordnet, antonyms_from_wordnet
from src.synonyms import synonyms_from_wordnet, contextual_synonyms
from src.embeddings import TransformerSynonymFinder
from src.llm_integration import predictive_text, generative_synonyms
from src.feedback import record_like, get_likes

def main():
    st.title("Advanced Dictionary–Thesaurus with LLM & Embeddings")
    st.write(
        "Combines classical WordNet lookups with transformer embeddings and "
        "LLM-based predictive text/generative synonyms."
    )

    # Ensure NLTK is initialized
    initialize_nltk_data()

    # Word input
    user_word = st.text_input("Enter a word to analyze:", value="fast")
    user_sentence = st.text_input("Enter a sentence for context (optional):", value="He likes to fast before a big race.")
    
    # Partial text for predictive completion
    partial_text = st.text_input("Predictive text (start writing something):", value="Today I want to")

    # On button click, do all dictionary–thesaurus tasks
    if st.button("Analyze"):
        if not user_word.strip():
            st.warning("Please enter a valid word.")
            return

        # 1. Definitions, examples, antonyms
        st.subheader("Definitions")
        defs = definitions_from_wordnet(user_word)
        if defs:
            for d in defs:
                st.write(f"- {d}")
        else:
            st.write("No definitions found.")

        st.subheader("Examples")
        exs = examples_from_wordnet(user_word)
        if exs:
            for e in exs:
                st.write(f"- {e}")
        else:
            st.write("No examples found.")

        st.subheader("Antonyms")
        ants = antonyms_from_wordnet(user_word)
        st.write(ants if ants else "No antonyms found.")

        # 2. Basic synonyms & contextual synonyms
        st.subheader("Basic Synonyms (WordNet)")
        basic_syns = synonyms_from_wordnet(user_word)
        if basic_syns:
            st.write(basic_syns)
        else:
            st.write("No synonyms found.")

        if user_sentence.strip():
            st.subheader("Contextual Synonyms (POS-based)")
            c_syns = contextual_synonyms(user_sentence, user_word)
            st.write(c_syns if c_syns else "No contextual synonyms found.")

        # 3. Transformer-based advanced synonyms
        st.subheader("Advanced Synonyms (Embedding-based)")
        finder = TransformerSynonymFinder()
        advanced = finder.rank_synonyms_by_context(user_word, user_sentence, top_n=5)
        if advanced:
            for syn in advanced:
                like_count = get_likes(user_word, syn)
                st.write(f"- {syn} (likes: {like_count})")
                if st.button(f"Like '{syn}'", key=syn):
                    record_like(user_word, syn)
                    st.experimental_rerun()  # So we see updated like count immediately
        else:
            st.write("No advanced synonyms found.")
    
    # 4. Predictive text (LLM-based)
    if st.button("Predict Next Words"):
        if partial_text.strip():
            completion = predictive_text(partial_text)
            st.subheader("Predictive Text Output")
            st.write(partial_text + completion)
        else:
            st.warning("Provide partial text to predict from.")

    # 5. Generative synonyms via LLM
    if st.button("Generate Synonyms"):
        if user_word.strip():
            gen_syn_output = generative_synonyms(user_word, user_sentence)
            st.subheader("LLM-based Generative Synonyms/Expressions")
            st.write(gen_syn_output)
        else:
            st.warning("Enter a valid word to generate synonyms.")

if __name__ == "__main__":
    main()
