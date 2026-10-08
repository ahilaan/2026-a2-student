"""One-pass unigram collection statistics with compact per-document counts.

Preparation analyzes each corpus record once to collect TF/DF, lengths and
packed uint32 term counts. Analyzed token sequences are retained when the
positional/pair scorer needs them. Candidate Counters are decoded lazily;
only the supplied candidate documents are scored. Older trusted local
statistics caches can still use the text-based fallback.
"""
import re
from array import array
from collections import Counter
from typing import Dict, List

_TOKEN_RE = re.compile(r"[a-z0-9]+")
ANALYZER = "stop_porter"


def tokenize(text: str) -> List[str]:
    """Apply the configured analyzer consistently to documents and queries.

    Raw alphanumeric mode remains available for hand examples.
    The shipped configuration removes stopwords and uses our own Porter port.
    """
    if ANALYZER == "raw":
        return _TOKEN_RE.findall(text.lower())
    from submission.analyzer import analyze
    return analyze(text, ANALYZER)


def candidate_rows(doc_ids, stats):
    """Shared per-call content records; pool order and duplicate semantics stay."""
    counts = stats.doc_term_counts
    lengths = stats.doc_lengths
    return [(doc, counts(doc), lengths[doc]) for doc in dict.fromkeys(doc_ids)]


class CollectionStats:
    """Collection statistics and packed document counts, decoded on demand."""

    def __init__(self) -> None:
        self.doc_texts: Dict[str, str] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.collection_term_counts: Counter = Counter()
        self.collection_length: int = 0
        self.doc_ids: List[str] = []
        self._term_counts_cache: Dict[str, Counter] = {}
        self.doc_freq: Counter = Counter()
        self._packed_counts = {}
        self._term_ids = {}
        self._terms = []
        self._collect_sequences = False
        self._packed_sequences = {}

    def add_document(self, doc_id: str, text: str) -> None:
        """Called once per document during from_corpus() -- tokenises
        every document exactly once, to build the collection-wide
        background model. Per-document Counters are NOT retained here
        (see doc_term_counts() below); only aggregate collection counts
        and the raw text (needed later if this doc_id turns out to be a
        candidate) are kept."""
        tokens = tokenize(text)
        counts = Counter(tokens)
        # Preserve the previous last-text-per-ID DF semantics even if the
        # corpus contains duplicate IDs; collection TF still counts records.
        if doc_id in self.doc_texts:
            previous = self.doc_term_counts(doc_id)
            for term in previous:
                self.doc_freq[term] -= 1
                if self.doc_freq[term] == 0:
                    del self.doc_freq[term]
            self._term_counts_cache.pop(doc_id, None)
        self.doc_freq.update(counts.keys())
        packed = array('I')
        ids, terms = self._term_ids, self._terms
        for term, frequency in counts.items():
            term_id = ids.get(term)
            if term_id is None:
                term_id = len(terms)
                ids[term] = term_id
                terms.append(term)
            packed.append(term_id)
            packed.append(frequency)
        # Two uint32s per unique term, rather than a Python Counter per doc.
        # Candidate Counters are decoded once on demand; raw text remains.
        self._packed_counts[doc_id] = packed.tobytes()
        if self._collect_sequences:
            # Four bytes per token; reuse prepare's analysis, never stem again.
            sequence = array('I', (ids[token] for token in tokens))
            self._packed_sequences[doc_id] = sequence.tobytes()
        self.doc_texts[doc_id] = text
        self.doc_lengths[doc_id] = len(tokens)
        self.collection_term_counts.update(counts)
        self.collection_length += len(tokens)
        self.doc_ids.append(doc_id)

    def doc_term_counts(self, doc_id: str) -> Counter:
        """Term counts for one document, decoded on first request and
        cached. Raises KeyError with a clear message for an unknown
        doc_id (e.g. a typo, or a doc_id from the wrong corpus)."""
        if doc_id not in self.doc_texts:
            raise KeyError(f"doc_id {doc_id!r} was not in the corpus passed to prepare()")
        if doc_id not in self._term_counts_cache:
            packed_counts = getattr(self, '_packed_counts', None)
            if packed_counts is not None and doc_id in packed_counts:
                packed = memoryview(packed_counts[doc_id]).cast('I')
                terms = self._terms
                self._term_counts_cache[doc_id] = Counter({terms[packed[i]]: packed[i+1]
                                                          for i in range(0, len(packed), 2)})
            else:
                # Backward compatibility for trusted older offline caches.
                self._term_counts_cache[doc_id] = Counter(tokenize(self.doc_texts[doc_id]))
        return self._term_counts_cache[doc_id]

    def collection_prob(self, term: str) -> float:
        """P(term | C), the background/collection language model used by
        both smoothing methods in Section 3.1. Returns 0.0 for an
        out-of-vocabulary term -- callers doing log-probability scoring
        must handle that (e.g. by flooring with a small epsilon), since
        log(0) is undefined."""
        if self.collection_length == 0:
            return 0.0
        return self.collection_term_counts.get(term, 0) / self.collection_length

    @classmethod
    def from_corpus(cls, corpus: List[tuple], collect_sequences=False) -> "CollectionStats":
        stats = cls()
        stats._collect_sequences = collect_sequences
        for doc_id, text in corpus:
            stats.add_document(doc_id, text)
        return stats


def dirichlet_smoothed_log_prob(term_count: int, doc_length: int, p_collection: float, mu: float) -> float:
    """log P(term | D) under Dirichlet-prior smoothing (assignment Section
    3.1): (c(term, D) + mu * P(term | C)) / (|D| + mu). Returns a very
    negative (not literally -inf) number if the numerator is 0, so a
    query-likelihood score built by summing this over query terms stays a
    finite, comparable float instead of collapsing to -inf on any single
    OOV term."""
    import math
    numerator = term_count + mu * p_collection
    denominator = doc_length + mu
    if denominator <= 0 or numerator <= 0:
        return -50.0  # floor, not -inf -- see docstring
    return math.log(numerator / denominator)


def jelinek_mercer_smoothed_log_prob(term_count: int, doc_length: int, p_collection: float, lam: float) -> float:
    """log P(term | D) under Jelinek-Mercer smoothing (assignment Section
    3.1): (1 - lam) * c(term, D)/|D| + lam * P(term | C). Same -50.0 floor
    convention as dirichlet_smoothed_log_prob() above, for the same reason."""
    import math
    doc_ml = term_count / doc_length if doc_length > 0 else 0.0
    prob = (1 - lam) * doc_ml + lam * p_collection
    if prob <= 0:
        return -50.0
    return math.log(prob)
