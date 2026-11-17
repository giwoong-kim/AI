"""RAG 핵심: 문단 분할 -> BM25 + 의미 검색 -> RRF 결합 -> cross-encoder 재정렬."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

EMBED_MODEL = os.environ.get("FAKEKILLER_EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("FAKEKILLER_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass
class Passage:
    text: str
    title: str = ""
    url: str = ""


# ---------- 문단 분할 ----------

_SENT_END = re.compile(r"(?<=[.!?。])\s+|\n+")


def chunk_text(text: str, max_chars: int = 600, overlap_sents: int = 1) -> list[str]:
    """문장 경계를 지키면서 max_chars 이하의 문단으로 자른다. 인접 문단은 문장 1개를 겹친다."""
    sents = [s.strip() for s in _SENT_END.split(text) if s and s.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for s in sents:
        if cur and size + len(s) > max_chars:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents else []
            size = sum(len(x) for x in cur)
        cur.append(s)
        size += len(s)
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c) >= 40]


# ---------- BM25 ----------

_WORD = re.compile(r"[a-z0-9]+|[가-힣]+")


def tokenize(text: str) -> list[str]:
    """영문·숫자는 단어 단위, 한글은 글자 bigram.

    한국어는 조사가 붙어 띄어쓰기 단위로는 '크레아틴이'와 '크레아틴'이 안 맞는다.
    형태소 분석기 없이 이를 완화하려고 bigram을 쓴다.
    """
    tokens: list[str] = []
    for w in _WORD.findall(text.lower()):
        if "가" <= w[0] <= "힣":
            tokens.extend([w] if len(w) == 1 else [w[i : i + 2] for i in range(len(w) - 1)])
        else:
            tokens.append(w)
    return tokens


# ---------- 모델 로딩 (처음 쓸 때 한 번) ----------


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL, max_length=512)


def embed(texts: list[str]) -> np.ndarray:
    return _embedder().encode(
        texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False
    )


# ---------- 결합 ----------


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: 점수 스케일이 다른 두 검색 결과를 순위만으로 합친다."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


# ---------- 검색기 ----------


class Retriever:
    """한 번의 검증에 쓰이는 문단 집합에 대한 인메모리 인덱스."""

    def __init__(self, passages: list[Passage], use_dense: bool = True):
        if not passages:
            raise ValueError("검색할 문단이 없습니다.")
        self.passages = passages
        self._bm25 = BM25Okapi([tokenize(f"{p.title} {p.text}") for p in passages])
        self._emb = embed([p.text for p in passages]) if use_dense else None

    def _bm25_rank(self, query: str, n: int) -> list[int]:
        scores = self._bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores)[:n]]

    def _dense_rank(self, query: str, n: int) -> list[int]:
        if self._emb is None:
            raise RuntimeError("use_dense=False 로 만든 인덱스입니다.")
        scores = self._emb @ embed([query])[0]
        return [int(i) for i in np.argsort(-scores)[:n]]

    def search(
        self, query: str, top_k: int = 5, candidates: int = 20, mode: str = "hybrid_rerank"
    ) -> list[tuple[Passage, float]]:
        """(문단, 점수) 목록을 관련도 순으로 돌려준다.

        점수는 mode에 따라 의미가 다르다. hybrid_rerank에서는 재정렬 모델의 관련도(0~1),
        나머지는 순위 기반 점수(1/순위)다.
        """
        if mode not in MODES:
            raise ValueError(f"mode는 {MODES} 중 하나여야 합니다.")
        n = min(candidates, len(self.passages))

        if mode == "bm25":
            ranked = self._bm25_rank(query, n)
        elif mode == "dense":
            ranked = self._dense_rank(query, n)
        else:
            ranked = rrf([self._bm25_rank(query, n), self._dense_rank(query, n)])[:n]

        if mode == "hybrid_rerank":
            scores = _reranker().predict(
                [(query, self.passages[i].text) for i in ranked], show_progress_bar=False
            )
            order = np.argsort(-np.asarray(scores))[:top_k]
            return [(self.passages[ranked[j]], float(scores[j])) for j in order]

        return [(self.passages[i], 1.0 / (r + 1)) for r, i in enumerate(ranked[:top_k])]
