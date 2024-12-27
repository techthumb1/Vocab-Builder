import torch
from sentence_transformers import SentenceTransformer, util
from .synonyms import synonyms_from_wordnet

class TransformerSynonymFinder:
    """
    Provides advanced, context-aware synonyms by ranking WordNet synonyms 
    using transformer embeddings.
    """
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def rank_synonyms_by_context(self, word: str, context: str, top_n: int = 5):
        """
        1) Collect synonyms from WordNet.
        2) Encode each with model.
        3) Compare to 'context' embedding, returning top_n by cosine similarity.
        """
        candidates = synonyms_from_wordnet(word)
        if not candidates:
            return []

        context_emb = self.model.encode(context, convert_to_tensor=True)
        results = []

        for c in candidates:
            c_emb = self.model.encode(c, convert_to_tensor=True)
            sim_score = util.cos_sim(context_emb, c_emb).item()
            results.append((c, sim_score))

        results.sort(key=lambda x: x[1], reverse=True)
        best_syns = [syn for syn, _ in results[:top_n]]
        return best_syns
