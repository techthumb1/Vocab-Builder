import nltk
from nltk.corpus import wordnet as wn


def definitions_from_wordnet(word: str):
    """
    Returns distinct definitions for 'word' from WordNet.
    """
    synsets = wn.synsets(word)
    unique_defs = {syn.definition() for syn in synsets}
    return list(unique_defs)

def examples_from_wordnet(word: str):
    """
    Returns usage examples for 'word' from WordNet.
    """
    synsets = wn.synsets(word)
    exs = []
    for syn in synsets:
        exs.extend(syn.examples())
    return list(set(exs))

def antonyms_from_wordnet(word: str):
    """
    Returns antonyms for 'word' if WordNet lemma data is present.
    """
    ants = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            if lemma.name().lower() == word.lower():
                for ant in lemma.antonyms():
                    ants.add(ant.name().replace('_', ' '))
    return list(ants)
