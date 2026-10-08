"""Ordered-pair Dirichlet query likelihood with a normalized independent prior.

P((u,v)|D)=(tf((u,v),D)+mu*P(u|C)*P(v|C))/(max(|D|-1,0)+mu).
The pair prior sums to one over the collection vocabulary squared. This is
an actual pair language model, not a proximity bonus or a rank fusion.
"""
import math
from collections import Counter, OrderedDict
from submission.lm_utils import tokenize

_CACHE = OrderedDict()
_SIZE = 0
_STATS_REF = None
MAX_CACHED_PAIRS = 100000


def clear():
    global _SIZE, _STATS_REF
    _CACHE.clear()
    _SIZE = 0
    _STATS_REF = None


def document_pairs(doc, stats):
    global _SIZE, _STATS_REF
    if _STATS_REF is not stats:
        clear()
        _STATS_REF = stats
    key = doc
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    packed = getattr(stats, '_packed_sequences', {}).get(doc)
    if packed is None:
        tokens = tokenize(stats.doc_texts[doc])
    else:
        tokens = [stats._terms[term_id] for term_id in memoryview(packed).cast('I')]
    counts = Counter(zip(tokens, tokens[1:]))
    if len(counts) <= MAX_CACHED_PAIRS:
        while _CACHE and (_SIZE + len(counts) > MAX_CACHED_PAIRS or len(_CACHE) >= 1024):
            _, old = _CACHE.popitem(last=False)
            _SIZE -= len(old)
        _CACHE[key] = counts
        _SIZE += len(counts)
    return counts


def interpolate(query, terms, rows, primary, stats, weight, mu):
    tokens = tokenize(query)
    pairs = Counter(zip(tokens, tokens[1:]))
    if not pairs or not weight:
        return primary
    backgrounds = [(pair, n, mu * stats.collection_prob(pair[0]) * stats.collection_prob(pair[1]))
                   for pair, n in pairs.items()]
    # Compare per-token likelihoods. For normalized RM3 the total mass is one;
    # for the original count query it is the query length, preserving its scale.
    scale = sum(terms.values()) / (2 * sum(pairs.values()))
    unigram = dict(primary)
    scores = []
    for doc, _, length in rows:
        counts = document_pairs(doc, stats)
        denominator = max(0, length - 1) + mu
        pair_score = sum(n * (math.log(numerator / denominator) if numerator > 0 else -100.)
                         for pair, n, background in backgrounds
                         for numerator in (counts.get(pair, 0) + background,))
        scores.append((doc, (1 - weight) * unigram[doc] + weight * scale * pair_score))
    scores.sort(key=lambda row: row[1], reverse=True)
    return scores
