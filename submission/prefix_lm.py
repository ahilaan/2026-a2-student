"""Normalized word probability mixture of full-document and leading-token LMs."""
import math
from collections import Counter, OrderedDict
from submission.lm_utils import tokenize

_CACHE = OrderedDict()
_STATS_REF = None
MAX_CACHED_DOCUMENTS = 1024


def clear():
    global _STATS_REF
    _CACHE.clear()
    _STATS_REF = None


def prefix_counts(doc, stats, span):
    global _STATS_REF
    if _STATS_REF is not stats:
        clear()
        _STATS_REF = stats
    key = (doc, span)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    packed = getattr(stats, '_packed_sequences', {}).get(doc)
    if packed is None:
        tokens = tokenize(stats.doc_texts[doc])[:span]
    else:
        tokens = [stats._terms[i] for i in memoryview(packed).cast('I')[:span]]
    result = Counter(tokens), len(tokens)
    while len(_CACHE) >= MAX_CACHED_DOCUMENTS:
        _CACHE.popitem(last=False)
    _CACHE[key] = result
    return result


def score(doc, counts, length, probabilities, stats, mu, weight, span, prefix_mu):
    prefix, n = prefix_counts(doc, stats, span)
    return sum(qw * (math.log(p) if p > 0 else -50.)
               for term, qw, pc in probabilities
               for p in ((1-weight)*(counts.get(term, 0)+mu*pc)/(length+mu)
                         + weight*(prefix.get(term, 0)+prefix_mu*pc)/(n+prefix_mu),))


def rank(terms, rows, k, stats, mu, weight, span, prefix_mu):
    probabilities = [(term, qw, stats.collection_prob(term)) for term, qw in terms.items()]
    scores = [(doc, score(doc, counts, length, probabilities, stats, mu, weight, span, prefix_mu))
              for doc, counts, length in rows]
    scores.sort(key=lambda row: row[1], reverse=True)
    return scores[:k]
