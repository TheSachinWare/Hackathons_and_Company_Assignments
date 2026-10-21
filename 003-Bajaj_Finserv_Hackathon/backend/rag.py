"""
rag.py
------
Grounding layer for general mutual-fund questions ("what's the expense ratio
of Flexi Cap", "what's the risk of the Gilt fund", "what's the fund's
objective").

Two-layer grounding, by design:
  1. STRUCTURED lookup against data/fund_facts.json first (see fund_facts.json
     docstring). Exact numbers ALWAYS come from here when available -- this
     is what actually prevents hallucinated AUM/expense-ratio figures, not
     the retrieval step.
  2. Narrative retrieval over chunked factsheet text, for qualitative
     questions ("what's the fund's investing philosophy", "what did the CIO
     say about markets") that aren't captured as structured fields.

Retrieval implementation note: this hackathon build uses a small,
dependency-free TF-IDF + cosine-similarity retriever (pure stdlib) so the
whole thing runs offline with no embedding-API dependency. In production
this file's `search()` function is the seam to swap in real embeddings
(e.g. text-embedding-3 / Voyage) + a managed vector store (pgvector,
OpenSearch k-NN, or a hosted FAISS/Chroma index) with zero changes needed
in the calling code -- see README "Production RAG" section.
"""

import json
import math
import re
from collections import Counter
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Chunk:
    doc_id: str
    text: str


class TfidfIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.doc_tokens = [_tokenize(c.text) for c in chunks]
        self.df = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] += 1
        self.n_docs = len(chunks)
        self.doc_vectors = [self._vectorize(toks) for toks in self.doc_tokens]

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n_docs + 1) / (df + 1)) + 1

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        length = max(len(tokens), 1)
        return {t: (c / length) * self._idf(t) for t, c in tf.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values())) or 1e-9
        norm_b = math.sqrt(sum(v * v for v in b.values())) or 1e-9
        return num / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        q_vec = self._vectorize(_tokenize(query))
        scored = [
            (self.chunks[i], self._cosine(q_vec, self.doc_vectors[i]))
            for i in range(self.n_docs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s in scored if s[1] > 0][:top_k]


def build_index_from_fund_facts(fund_facts: dict) -> TfidfIndex:
    """
    Turns each fund's structured facts into a small narrative chunk, so the
    same retrieval code path serves both 'give me the exact expense ratio'
    (structured lookup, see fact_lookup()) and 'tell me about this fund'
    (retrieval) queries. In production this is replaced/extended by chunking
    the actual factsheet PDF pages (MD letter, CIO letter, macro outlook,
    per-fund pages) -- kept out of this offline demo to avoid re-deriving
    the whole PDF text pipeline inside the sandbox.
    """
    chunks = []
    for f in fund_facts["funds"]:
        text = (
            f"{f['scheme_name']} is a {f['category']} ({f['type']}) fund. "
            f"Objective: {f.get('objective_summary','')} "
            f"Benchmark: {f['benchmark']}. Inception date: {f['inception_date']}. "
            f"AUM: Rs {f['aum_cr']} crore. Regular plan expense ratio: {f['expense_ratio_regular_pct']}%. "
            f"Direct plan expense ratio: {f['expense_ratio_direct_pct']}%. "
            f"Riskometer: {f['riskometer_scheme']}. Minimum investment: Rs {f['min_investment_inr']}."
        )
        chunks.append(Chunk(doc_id=f["scheme_name"], text=text))
    return TfidfIndex(chunks)


def fact_lookup(fund_facts: dict, scheme_name_query: str) -> dict | None:
    """Exact/fuzzy structured lookup -- preferred path for numeric questions."""
    q = scheme_name_query.lower()
    best, best_score = None, 0
    for f in fund_facts["funds"]:
        name = f["scheme_name"].lower()
        score = sum(1 for tok in _tokenize(q) if tok in name)
        if score > best_score:
            best, best_score = f, score
    return best if best_score > 0 else None


def load_fund_facts(path: str = "../data/fund_facts.json") -> dict:
    with open(path, "r") as fh:
        return json.load(fh)
