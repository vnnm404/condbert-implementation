import numpy as np

from flair.data import Sentence
from flair.embeddings import WordEmbeddings

from helper.utils import calculate_cosine_similarity

class TextEmbedder:
    def __init__(self):
        self.embedding = WordEmbeddings("glove")
    
    def get_embedding(self, text):
        tokens = self.embedding.embed(Sentence(text))[0]
        return np.mean([token.embedding.cpu().numpy() for token in tokens], axis=0) if tokens else np.zeros(self.embedding.embedding_length)

class TextDecoder:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def decode(self, tokens):
        return tokens if isinstance(tokens, str) else self.tokenizer.convert_tokens_to_string(tokens) if self.tokenizer else " ".join(tokens).replace(" ##", "")

class EmbeddingSimilarity:
    def __init__(self, sim_coef=100, tokenizer = None):
        self.embedder = TextEmbedder()
        self.decoder = TextDecoder(tokenizer = tokenizer)
        self.similarity_coefficient = sim_coef

    def rank_hypotheses(self, original_embedding, hypotheses, scores):
        candidates = [
            (hypothesis, score, calculate_cosine_similarity(original_embedding, self.embedder.get_embedding(self.decoder.decode(hypothesis))))
            for hypothesis, score in zip(hypotheses, scores)
        ]
        return sorted(candidates, key=lambda x: x[1] + self.similarity_coefficient * x[2], reverse=True)

    def choose_best_hypothesis(self, hypotheses, original, scores):
        original_embedding = self.embedder.get_embedding(self.decoder.decode(original))
        ranked_candidates = self.rank_hypotheses(original_embedding, hypotheses, scores)
        return ranked_candidates[0][0]

    def __call__(self, **kwargs):
        return self.choose_best_hypothesis(kwargs['hypotheses'], kwargs['original'], kwargs['scores'])
