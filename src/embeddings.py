import torch
from sentence_transformers import SentenceTransformer, util
from .synonyms import synonyms_from_wordnet

class TransformerSynonymFinder:
    """
    Provides advanced, context-aware synonyms by ranking synonyms 
    using transformer embeddings.
    """
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def rank_synonyms(self, synonyms, context: str, top_n: int = 5):
        """
        Ranks a list of 'synonyms' by semantic similarity to 'context'.
        """
        if not synonyms:
            return []
        if not context.strip():
            # If no context is provided, just return the first top_n
            return synonyms[:top_n]

        context_emb = self.model.encode(context, convert_to_tensor=True)
        results = []
        for syn in synonyms:
            syn_emb = self.model.encode(syn, convert_to_tensor=True)
            similarity = util.cos_sim(context_emb, syn_emb).item()
            results.append((syn, similarity))

        # Sort by descending similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return [syn for syn, _ in results[:top_n]]

    def rank_synonyms_by_context(self, word: str, context: str, top_n: int = 5):
        """
        1) Collect synonyms from WordNet for 'word'.
        2) Calls rank_synonyms(...) internally to rank them by similarity to 'context'.
        """
        candidates = synonyms_from_wordnet(word)
        if not candidates:
            return []
        return self.rank_synonyms(candidates, context, top_n=top_n)
