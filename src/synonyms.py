import nltk
import os
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize

# Set NLTK data path and download required data
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

for resource in ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True, download_dir=nltk_data_dir)
def _wordnet_pos(treebank_tag: str):
    """
    Maps POS tags from Treebank to WordNet's format.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return None

def synonyms_from_wordnet(word: str, pos_filter=None):
    """
    Returns synonyms for 'word'. If pos_filter (like wordnet.VERB) is given, 
    synonyms are restricted to that part of speech.
    """
    syns = set()
    for synset in wordnet.synsets(word):
        if pos_filter and synset.pos() != pos_filter:
            continue
        for lemma in synset.lemmas():
            if lemma.name().lower() != word.lower():
                syns.add(lemma.name().replace('_', ' '))
    return list(syns)

def contextual_synonyms(sentence: str, target_word: str):
    """
    Uses POS tagging on 'sentence' to find synonyms of 'target_word'
    that match its usage. If no match, returns general synonyms.
    """
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    wn_pos = None

    for tok, tag in tagged:
        if tok.lower() == target_word.lower():
            wn_pos = _wordnet_pos(tag)
            break

    if wn_pos:
        return synonyms_from_wordnet(target_word, pos_filter=wn_pos)
    else:
        return synonyms_from_wordnet(target_word)
