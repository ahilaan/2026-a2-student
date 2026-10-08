import math
from collections import Counter
import pytest
from submission import feedback as f, lm_utils as lm, prefix_lm as p

@pytest.fixture
def stats(monkeypatch):
    monkeypatch.setattr(lm, 'ANALYZER', 'raw')
    s = lm.CollectionStats.from_corpus([('a', 'alpha alpha beta gamma'), ('b', 'beta beta alpha gamma'), ('tie', 'alpha alpha beta gamma'), ('empty', '')], collect_sequences=True)
    monkeypatch.setattr(f, '_STATS', s)
    monkeypatch.setattr(f, '_DOC_FREQ', s.doc_freq)
    monkeypatch.setattr(f, '_NUM_DOCS', len(s.doc_texts))
    monkeypatch.setattr(f, 'PREFIX_LM_WEIGHT', 0.25)
    monkeypatch.setattr(f, 'DIRICHLET_MU', 400.0)
    monkeypatch.setattr(f, 'PREFIX_LM_TOKENS', 2)
    monkeypatch.setattr(f, 'PREFIX_LM_MU', 100.0)
    monkeypatch.setattr(f, 'BIGRAM_LM_WEIGHT', 0.15)
    p.clear()
    return s

def test_normalized_probability_and_independent_score(stats):
    for doc in stats.doc_texts:
        full, length = (stats.doc_term_counts(doc), stats.doc_lengths[doc])
        lead, n = p.prefix_counts(doc, stats, 2)
        probs = {t: 0.75 * (full.get(t, 0) + 400 * stats.collection_prob(t)) / (length + 400) + 0.25 * (lead.get(t, 0) + 100 * stats.collection_prob(t)) / (n + 100) for t in stats.collection_term_counts}
        assert sum(probs.values()) == pytest.approx(1.0)
        expected = 2 * math.log(probs['alpha']) + math.log(probs['beta']) - 50.0
        actual = f._doc_ql_score(doc, {'alpha': 2, 'beta': 1, 'unseen': 1}, stats)
        assert actual == pytest.approx(expected)
        single = f._rank_prepared({'alpha': 2, 'beta': 1, 'unseen': 1}, [(doc, full, length)], 1, stats)
        assert single == [(doc, actual)]

@pytest.mark.parametrize('query', ['alpha beta', 'alpha alpha beta', 'alpha unseen', 'unseen', ''])
@pytest.mark.parametrize('weight', [0.1, 0.25, 0.5])
def test_feedback_disabled_exact_float_bits_and_ties(stats, monkeypatch, query, weight):
    monkeypatch.setattr(f, 'PREFIX_LM_WEIGHT', weight)
    monkeypatch.setattr(f, 'RM3_LAMBDA_ORIG', 1.0)
    pool = ['tie', 'a', 'b', 'empty', 'tie']
    a, b = (f.score_candidates(query, pool), f.relevance_model_feedback(query, ['a', 'b'], pool))
    assert [(d, s.hex()) for d, s in a] == [(d, s.hex()) for d, s in b]
    if query:
        assert [d for d, _ in a].index('tie') < [d for d, _ in a].index('a')

def test_packed_prefix_last_duplicate_and_cache_bounds(monkeypatch):
    monkeypatch.setattr(lm, 'ANALYZER', 'raw')
    records = [('a', 'old old'), ('b', ''), ('a', 'alpha beta gamma')]
    raw, packed = (lm.CollectionStats.from_corpus(records), lm.CollectionStats.from_corpus(records, collect_sequences=True))
    expected = {doc: p.prefix_counts(doc, raw, 2) for doc in raw.doc_texts}
    monkeypatch.setattr(p, 'tokenize', lambda text: pytest.fail('Packed prefix reanalyzed'))
    monkeypatch.setattr(p, 'MAX_CACHED_DOCUMENTS', 1)
    for doc in packed.doc_texts:
        assert p.prefix_counts(doc, packed, 2) == expected[doc]
        assert len(p._CACHE) <= 1
    assert expected['a'] == (Counter({'alpha': 1, 'beta': 1}), 2)

def test_zero_weight_uses_original_ql_arithmetic(stats, monkeypatch):
    monkeypatch.setattr(f, 'PREFIX_LM_WEIGHT', 0.0)
    monkeypatch.setattr(f, 'BIGRAM_LM_WEIGHT', 0.0)
    terms = {'alpha': 2, 'beta': 1, 'unseen': 1}
    pool = ['tie', 'a', 'b', 'empty']
    expected = sorted([(d, sum((w * lm.dirichlet_smoothed_log_prob(stats.doc_term_counts(d).get(t, 0), stats.doc_lengths[d], stats.collection_prob(t), f.DIRICHLET_MU) for t, w in terms.items()))) for d in pool], key=lambda row: row[1], reverse=True)
    assert f.score_candidates('alpha alpha beta unseen', pool) == expected
