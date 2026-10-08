"""Shared unigram/prefix/bigram query likelihood and robust RM1 -> RM3 feedback.

The supplied seed is the sole feedback evidence, including documents outside
our candidate pool. No query-specific ranking or clean-seed state is retained.
RM2 is independently callable; the competition ranking uses RM1 and RM3.
"""
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

from submission.corpus_utils import iter_corpus
from submission.lm_utils import CollectionStats, dirichlet_smoothed_log_prob, tokenize, candidate_rows

# Final collection-wide settings; no query-specific rules.
DIRICHLET_MU = 500.0
RM_DOC_MU = 250.0
RM3_LAMBDA_ORIG = 0.88
FEEDBACK_TERMS = 15
MIN_SUPPORT_FRAC = 0.20
SEED_POSTERIOR_TEMPERATURE = 1.0
QUERY_COVERAGE_FLOOR = 0.25
ANCHOR_EXPONENT = 1.0
CONFIDENCE_FLOOR = 0.4
BIGRAM_LM_WEIGHT = 0.1
BIGRAM_LM_MU = 500.0
PREFIX_LM_WEIGHT = 0.25
PREFIX_LM_TOKENS = 32
PREFIX_LM_MU = 100.0


# Expansion-only filtering: base QL and the original query retain all tokens.
_STOPWORDS = frozenset("""
a an the and or but if then else as at by for from in into of on onto to with
without about above after again against all am are be because been before being
below between both can could did do does doing down during each few further had
has have having he her here hers herself him himself his how i is it its itself
just me more most my myself no nor not now off once only other our ours ourselves
out over own same she should so some such than that their theirs them themselves
there these they this those through too under until up very was we were what when
where which while who whom why will would you your yours yourself yourselves
also may might must s t d ll re ve don doesn isn aren wasn weren
""".split())

_STATS: Optional[CollectionStats] = None
_DOC_FREQ: Counter = Counter()
_NUM_DOCS = 0


def prepare(corpus_path: str) -> None:
    """Build collection statistics and compact analyzed sequences once."""
    global _STATS, _DOC_FREQ, _NUM_DOCS
    stats = CollectionStats.from_corpus(iter_corpus(corpus_path),
        collect_sequences=bool(BIGRAM_LM_WEIGHT or PREFIX_LM_WEIGHT))
    _STATS, _DOC_FREQ, _NUM_DOCS = stats, stats.doc_freq, len(stats.doc_texts)
    from submission import bigram_lm, prefix_lm
    bigram_lm.clear()
    prefix_lm.clear()


def _require_stats() -> CollectionStats:
    if _STATS is None:
        raise RuntimeError("Call prepare(corpus_path) before retrieval.")
    return _STATS


def _query_counter(query: str) -> Counter:
    return Counter(tokenize(query))


def _doc_ql_score(doc_id: str, terms: Dict[str, float], stats: CollectionStats,
                  mu: Optional[float] = None) -> float:
    if mu is None:
        mu = DIRICHLET_MU
    counts = stats.doc_term_counts(doc_id)
    length = stats.doc_lengths[doc_id]
    if PREFIX_LM_WEIGHT:
        from submission.prefix_lm import score
        probabilities = [(term, weight, stats.collection_prob(term)) for term, weight in terms.items()]
        return score(doc_id, counts, length, probabilities, stats, mu,
                     PREFIX_LM_WEIGHT, PREFIX_LM_TOKENS, PREFIX_LM_MU)
    # OOV terms contribute the same finite floor to every document.
    return sum(weight * dirichlet_smoothed_log_prob(
        counts.get(term, 0), length, stats.collection_prob(term), mu
    ) for term, weight in terms.items())


def _rank_with_term_distribution(terms: Dict[str, float], doc_ids: List[str],
                                 k: int, stats: CollectionStats) -> List[Tuple[str, float]]:
    if not terms or k <= 0:
        return []
    return _rank_prepared(terms, candidate_rows(doc_ids, stats), k, stats)


def _rank_prepared(terms, rows, k, stats, use_prefix=True):
    """Dirichlet QL arithmetic; collection probabilities reused per query."""
    if use_prefix and PREFIX_LM_WEIGHT:
        from submission.prefix_lm import rank
        return rank(terms, rows, k, stats, DIRICHLET_MU,
                    PREFIX_LM_WEIGHT, PREFIX_LM_TOKENS, PREFIX_LM_MU)
    probabilities = [(term, weight, stats.collection_prob(term)) for term, weight in terms.items()]
    log = math.log
    mu = DIRICHLET_MU
    backgrounds = [(term, weight, mu * pc) for term, weight, pc in probabilities]
    def score(counts, length):
        denominator = length + mu
        get = counts.get
        return sum(weight * (-50.0 if denominator <= 0 or numerator <= 0 else log(numerator / denominator))
                   for term, weight, background in backgrounds
                   for numerator in (get(term, 0) + background,))
    scores = [(doc, score(counts, length)) for doc, counts, length in rows]
    # Stable sort retains candidate-pool order on exact ties.
    scores.sort(key=lambda pair: pair[1], reverse=True)
    return scores[:k]


def unigram_score_candidates(query: str, candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]:
    """Independently testable, exact unigram QL, with no rank/positional signals."""
    terms, stats = _query_counter(query), _require_stats()
    if not terms or k <= 0:
        return []
    return _rank_prepared(terms, candidate_rows(candidate_doc_ids, stats), k, stats, use_prefix=False)


def _competition_rank(query: str, terms: Dict[str, float], doc_ids: List[str],
                       k: int, stats: CollectionStats) -> List[Tuple[str, float]]:
    """Shared normalized word/pair query-likelihood scorer for every track."""
    if not terms or k <= 0:
        return []
    ids = list(dict.fromkeys(doc_ids))
    counts, lengths = stats.doc_term_counts, stats.doc_lengths
    rows = [(doc, counts(doc), lengths[doc]) for doc in ids]
    primary = _rank_prepared(terms, rows, len(ids), stats)
    if BIGRAM_LM_WEIGHT:
        from submission.bigram_lm import interpolate
        primary = interpolate(query, terms, rows, primary, stats, BIGRAM_LM_WEIGHT, BIGRAM_LM_MU)
    return primary[:k]


def score_candidates(query: str, candidate_doc_ids: List[str], k: int = 10) -> List[Tuple[str, float]]:
    """Shared full-document/prefix word QL with an ordered-bigram LM."""
    query_counts = _query_counter(query)
    return _competition_rank(query, query_counts, candidate_doc_ids,
                             k, _require_stats())


def _idf(term: str) -> float:
    df = _DOC_FREQ.get(term, 0)
    return math.log1p((_NUM_DOCS - df + 0.5) / (df + 0.5))


def _prepare_coverage(query_counts: Counter, stats: CollectionStats):
    weights = {term: _idf(term) for term in query_counts
               if stats.collection_prob(term) > 0}
    return weights, sum(weights.values())


def _prepared_coverage(prepared, doc_id: str, stats: CollectionStats) -> float:
    weights, total = prepared
    if total == 0:
        return 0.0
    counts = stats.doc_term_counts(doc_id)
    return sum(weight for term, weight in weights.items() if counts.get(term, 0)) / total




def _usable_seed(doc_ids: List[str], stats: CollectionStats) -> List[str]:
    # Missing/empty documents contain no evidence. Never substitute other docs.
    return [doc_id for doc_id in dict.fromkeys(doc_ids)
            if stats.doc_lengths.get(doc_id, 0) > 0]


def _seed_posteriors(query_counts: Counter, seed: List[str],
                     stats: CollectionStats, coverage=None) -> Dict[str, float]:
    if not seed:
        return {}
    divisor = max(1, sum(query_counts.values()))
    scores = [_doc_ql_score(doc_id, query_counts, stats) / divisor for doc_id in seed]
    maximum = max(scores)
    if coverage is None:
        prepared = _prepare_coverage(query_counts, stats)
        coverage = {doc: _prepared_coverage(prepared, doc, stats) for doc in seed}
    raw = {}
    for doc_id, score in zip(seed, scores):
        trust = QUERY_COVERAGE_FLOOR + (1 - QUERY_COVERAGE_FLOOR) * coverage[doc_id]
        raw[doc_id] = math.exp((score - maximum) / SEED_POSTERIOR_TEMPERATURE) * trust
    total = sum(raw.values())
    return {doc_id: value / total for doc_id, value in raw.items()}


def _estimate_rm1(posteriors: Dict[str, float], stats: CollectionStats) -> Tuple[Dict[str, float], Counter]:
    """Exact smoothed RM1 on the seed vocabulary, plus document support.

    R(w) = sum_D posterior(D)*tf(w,D)/(|D|+mu)
           + P(w|C)*sum_D posterior(D)*mu/(|D|+mu).
    The second term includes EVERY seed doc, even when w is absent.
    The full RM1 also has background mass outside the returned vocabulary;
    selection needs only terms actually observed in the seed.
    """
    observed, support = Counter(), Counter()
    background = 0.0
    for doc_id, posterior in posteriors.items():
        counts = stats.doc_term_counts(doc_id)
        denominator = stats.doc_lengths[doc_id] + RM_DOC_MU
        background += posterior * RM_DOC_MU / denominator
        for term, count in counts.items():
            observed[term] += posterior * count / denominator
            support[term] += 1
    return ({term: mass + stats.collection_prob(term) * background
             for term, mass in observed.items()}, support)


def _estimate_rm2(query_counts: Counter, seed: List[str], stats: CollectionStats) -> Dict[str, float]:
    """Conditional-independence RM2, normalized over the seed vocabulary.

    With uniform P(D) over F: P_F(w)=sum_D P(w|D)/|F|,
    P(D|w)=P(w|D)/sum_E P(w|E), and
    R2(w) proportional to P_F(w)*product_q [sum_D P(q|D)P(D|w)]^qtf(q).
    Query terms are independent conditional on w (rather than D as in RM1).
    Use log-space; OOV query factors are constant and omitted.
    """
    seed = _usable_seed(seed, stats)
    if not seed:
        return {}
    counts = [stats.doc_term_counts(doc_id) for doc_id in seed]
    denominators = [stats.doc_lengths[doc_id] + RM_DOC_MU for doc_id in seed]
    vocabulary = sorted(set().union(*(set(count) for count in counts)))

    def probabilities(term):
        background = RM_DOC_MU * stats.collection_prob(term)
        return [(count.get(term, 0) + background) / denominator
                for count, denominator in zip(counts, denominators)]

    query_probs = {term: probabilities(term) for term in query_counts
                   if stats.collection_prob(term) > 0}
    log_masses = {}
    for term in vocabulary:
        word_probs = probabilities(term)
        total = sum(word_probs)
        value = math.log(total / len(seed))
        for query_term, probs in query_probs.items():
            conditional = sum(pq * pw for pq, pw in zip(probs, word_probs)) / total
            value += query_counts[query_term] * math.log(conditional)
        log_masses[term] = value
    maximum = max(log_masses.values())
    raw = {term: math.exp(value - maximum) for term, value in log_masses.items()}
    total = sum(raw.values())
    return {term: value / total for term, value in raw.items()}


def _build_rm3_model(query_counts: Counter, selected: Dict[str, float],
                     evidence_confidence: float = 1.0) -> Dict[str, float]:
    total = sum(query_counts.values())
    if total == 0:
        return {}
    confidence = min(1.0, len(selected) / FEEDBACK_TERMS)
    original_mass = 1 - (1 - RM3_LAMBDA_ORIG) * confidence * evidence_confidence if selected else 1.0
    model = {term: original_mass * count / total for term, count in query_counts.items()}
    for term, probability in selected.items():
        model[term] = model.get(term, 0.0) + (1 - original_mass) * probability
    return model


def _feedback_evidence(query_counts: Counter, seed: List[str], stats: CollectionStats,
                       diagnostics=True) -> Dict:
    """Seed-only confidence and term selection, with auditable diagnostics.

    Query terms may keep their RM1 mass; novel terms are ranked by raw
    RM1 probability with anchoring and support anchored to query-covering seed docs. Confidence
    uses coverage mean/lower quartile, effective sample size and term agreement.
    None of these quantities uses judgments or a cached clean reference set.
    """
    from submission.analyzer import STOPWORDS
    if not seed or not query_counts:
        return {"model": {}, "selected": {}, "confidence": 0.0, "effective_lambda": 1.0}
    prepared = _prepare_coverage(query_counts, stats)
    query_coverage = {doc: _prepared_coverage(prepared, doc, stats) for doc in seed}
    posterior = _seed_posteriors(query_counts, seed, stats, query_coverage)
    content_query = Counter({term: count for term, count in query_counts.items()
                             if term not in STOPWORDS})
    # Raw/Porter modes can retain stopwords. Reuse only identical ordered terms.
    if list(content_query.items()) == list(query_counts.items()):
        coverage = query_coverage
    else:
        prepared = _prepare_coverage(content_query, stats)
        coverage = {doc: _prepared_coverage(prepared, doc, stats) for doc in seed}
    rm1, support = _estimate_rm1(posterior, stats)
    anchors = {}
    for doc in seed:
        for term in stats.doc_term_counts(doc):
            anchors[term] = anchors.get(term, 0.0) + posterior[doc] * coverage[doc]
    occurrence = Counter()
    for doc in seed:
        for term in stats.doc_term_counts(doc):
            occurrence[term] += posterior[doc]
    anchors = {term: value / occurrence[term] for term, value in anchors.items()}
    minimum = max(2 if len(seed) >= 2 else 1, math.ceil(MIN_SUPPORT_FRAC * len(seed)))
    candidates = [term for term in rm1 if term in query_counts or
                  (len(term) > 1 and not term.isdigit() and term not in _STOPWORDS
                   and support[term] >= minimum)]

    def selection_score(term):
        value = rm1[term]
        return value * (0.15 + anchors[term]) ** ANCHOR_EXPONENT

    candidates.sort(key=lambda term: (-selection_score(term), term))
    # Original terms retain their relevance mass and consume the same budget.
    originals = [term for term in candidates if term in query_counts]
    novel = [term for term in candidates if term not in query_counts and selection_score(term) > 0]
    chosen = originals + novel[:max(0, FEEDBACK_TERMS - len(originals))]
    total = sum(rm1[term] for term in chosen)
    selected = {term: rm1[term] / total for term in chosen} if total else {}
    values = sorted(coverage.values())
    lower_quartile = values[int((len(values) - 1) * 0.25)]
    mean_coverage = sum(posterior[doc] * coverage[doc] for doc in seed)
    ess = 1 / sum(value * value for value in posterior.values())
    anchored_agreement = sum(probability * anchors[term] for term, probability in selected.items())
    confidence = mean_coverage * math.sqrt(lower_quartile * ess / len(seed)) * anchored_agreement
    confidence = min(1.0, confidence)
    novel_terms = [term for term in selected if term not in query_counts]
    if confidence < CONFIDENCE_FLOOR or not novel_terms:
        confidence = 0.0
    model = _build_rm3_model(query_counts, selected, confidence)
    if not diagnostics:
        return {"model": model, "confidence": confidence}
    agreement = sum(probability * support[term] / len(seed) for term, probability in selected.items())
    size_confidence = min(1.0, len(selected) / FEEDBACK_TERMS)
    effective_lambda = 1 - (1 - RM3_LAMBDA_ORIG) * size_confidence * confidence * 1.0
    return {"model": model, "selected": selected, "effective_lambda": effective_lambda,
            "confidence": confidence, "mean_coverage": mean_coverage,
            "coverage_lower_quartile": lower_quartile, "posterior_ess": ess,
            "posterior_entropy": -sum(p * math.log(p) for p in posterior.values() if p),
            "term_agreement": agreement, "anchored_agreement": anchored_agreement,
            "seed": [{"doc_id": doc, "posterior": posterior[doc], "coverage": coverage[doc],
                      "ql": _doc_ql_score(doc, query_counts, stats)} for doc in seed],
            "terms": [{"term": term, "weight": probability, "rm1": rm1[term],
                       "support": support[term], "anchored_coverage": anchors[term]}
                      for term, probability in selected.items()]}


def feedback_diagnostics(query: str, pseudo_relevant_doc_ids: List[str]) -> Dict:
    """Optional offline diagnostic helper; does not rank or inspect judgments."""
    stats = _require_stats()
    return _feedback_evidence(_query_counter(query), _usable_seed(pseudo_relevant_doc_ids, stats), stats)


def relevance_model_feedback(
    query: str,
    pseudo_relevant_doc_ids: List[str],
    candidate_doc_ids: List[str],
    k: int = 10,
) -> List[Tuple[str, float]]:
    """Estimate robust RM1 from the given seed, interpolate RM3, rerank pool."""
    stats = _require_stats()
    query_counts = _query_counter(query)
    if RM3_LAMBDA_ORIG >= 1.0:
        return _competition_rank(query, query_counts, candidate_doc_ids, k, stats)
    seed = _usable_seed(pseudo_relevant_doc_ids, stats)
    if not query_counts or not seed or k <= 0:
        return _competition_rank(query, query_counts, candidate_doc_ids, k, stats)
    evidence = _feedback_evidence(query_counts, seed, stats, diagnostics=False)
    if evidence["confidence"] == 0 or RM3_LAMBDA_ORIG >= 1.0:
        return _competition_rank(query, query_counts, candidate_doc_ids, k, stats)
    model = evidence["model"]
    return _competition_rank(query, model, candidate_doc_ids, k, stats)
