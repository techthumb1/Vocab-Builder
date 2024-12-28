import sentence_transformers
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.add(lemma.name())
    return list(synonyms)

class TransformerSynonymFinder:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def rank_synonyms(self, synonyms, context: str, top_n: int = 5):
        if not synonyms or not context.strip():
            return synonyms[:top_n] if synonyms else []

        context_emb = self.model.encode(context, convert_to_tensor=True)
        results = []
        for syn in synonyms:
            syn_emb = self.model.encode(syn, convert_to_tensor=True)
            similarity = util.cos_sim(context_emb, syn_emb).item()
            results.append((syn, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return [syn for syn, _ in results[:top_n]]

    def rank_synonyms_by_context(self, word: str, context: str, top_n: int = 5):
        candidates = get_synonyms(word)
        return self.rank_synonyms(candidates, context, top_n=top_n)