"""Local text embeddings (unit-normalised), so cosine similarity is a dot product.

- ``SentenceTransformerEmbedder``: the real backend (spec §4.1), runs locally.
- ``HashingEmbedder``: deterministic hashing of words + character trigrams; no model download.
  Used for tests and offline runs. Lexical overlap stands in for semantic similarity.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Protocol

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9']+")
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with", "my", "i", "me",
    "was", "were", "is", "it", "that", "this", "we", "us", "by", "from", "as", "be", "had", "has",
    "but", "so", "then", "about", "into", "her", "his", "their", "our", "she", "he", "they", "who",
}


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)


class HashingEmbedder:
    def __init__(self, dim: int, ngram_weight: float):
        self.dim = dim
        self.ngram_weight = ngram_weight
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[tuple[str, float]]:
        toks = [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
        # Whole words, plus character trigrams so related word forms ("walked"/"walking")
        # and partial overlaps give graded rather than all-or-nothing similarity.
        feats = [(f"w:{t}", 1.0) for t in toks]
        for t in toks:
            padded = f"<{t}>"
            feats += [(f"c:{padded[k:k + 3]}", self.ngram_weight) for k in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feat, weight in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += weight if (h >> 63) & 1 else -weight
        return _normalise(out)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        self.name = f"st:{model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def make_embedder(cfg: dict[str, Any], backend: str | None = None) -> Embedder:
    e = cfg["embedding"]
    backend = backend or e["backend"]
    if backend == "hashing":
        return HashingEmbedder(e["hashing_dim"], e["hashing_ngram_weight"])
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(e["model"])
    raise ValueError(f"Unknown embedding backend: {backend}")
