import math
import pytest
from submission import feedback as f, lm_utils as lm, bigram_lm as b

@pytest.fixture
def corpus(monkeypatch):
    monkeypatch.setattr(lm, 'ANALYZER', 'raw')
    stats = lm.CollectionStats.from_corpus([('a', 'alpha beta alpha'), ('b', 'beta alpha alpha'), ('tie', 'alpha beta alpha'), ('empty', '')])
    monkeypatch.setattr(f, '_STATS', stats)
    monkeypatch.setattr(f, '_DOC_FREQ', stats.doc_freq)
    monkeypatch.setattr(f, '_NUM_DOCS', 4)
    monkeypatch.setattr(f, 'BIGRAM_LM_WEIGHT', 0.15)
    b.clear()
    return stats

def test_pair_distribution_normalizes(corpus):
    for doc in corpus.doc_texts:
        counts = b.document_pairs(doc, corpus)
        total = sum(((counts.get((u, v), 0) + 500 * corpus.collection_prob(u) * corpus.collection_prob(v)) / (max(0, corpus.doc_lengths[doc] - 1) + 500) for u in corpus.collection_term_counts for v in corpus.collection_term_counts))
        assert total == pytest.approx(1)

@pytest.mark.parametrize('weight', [0.05, 0.15, 0.25])
def test_scalar_score_and_tie_order(corpus, monkeypatch, weight):
    monkeypatch.setattr(f, 'BIGRAM_LM_WEIGHT', weight)
    pool = ['tie', 'a', 'b', 'empty', 'a']
    actual = f.score_candidates('alpha beta', pool, 10)
    for doc, score in actual:
        unigram = f._doc_ql_score(doc, {'alpha': 1, 'beta': 1}, corpus)
        pair = b.document_pairs(doc, corpus).get(('alpha', 'beta'), 0) + 500 * corpus.collection_prob('alpha') * corpus.collection_prob('beta')
        expected = (1 - weight) * unigram + weight * math.log(pair / (max(0, corpus.doc_lengths[doc] - 1) + 500))
        assert score == expected
    assert [d for d, _ in actual].index('tie') < [d for d, _ in actual].index('a')

@pytest.mark.parametrize('q', ['alpha beta', 'alpha alpha beta', 'alpha', 'alpha unseen', ''])
@pytest.mark.parametrize('weight', [0.05, 0.15, 0.25])
def test_exact_disabled_feedback(corpus, monkeypatch, q, weight):
    monkeypatch.setattr(f, 'BIGRAM_LM_WEIGHT', weight)
    monkeypatch.setattr(f, 'RM3_LAMBDA_ORIG', 1.0)
    pool = ['tie', 'a', 'b', 'empty', 'tie']
    assert f.score_candidates(q, pool, 10) == f.relevance_model_feedback(q, ['a', 'b'], pool, 10)

def test_bounded_cache(corpus, monkeypatch):
    monkeypatch.setattr(b, 'MAX_CACHED_PAIRS', 2)
    for doc in corpus.doc_texts:
        b.document_pairs(doc, corpus)
    assert b._SIZE <= 2

def test_packed_sequence_equivalence_duplicates_and_no_query_reanalysis(monkeypatch):
    monkeypatch.setattr(lm, 'ANALYZER', 'raw')
    records = [('a', 'alpha beta alpha'), ('b', ''), ('a', 'beta alpha beta')]
    serial = lm.CollectionStats.from_corpus(records)
    packed = lm.CollectionStats.from_corpus(records, collect_sequences=True)
    for attr in ['collection_length', 'collection_term_counts', 'doc_freq', 'doc_lengths', 'doc_ids', '_packed_counts', '_term_ids', '_terms']:
        assert getattr(serial, attr) == getattr(packed, attr)
    expected = {doc: b.document_pairs(doc, serial) for doc in serial.doc_texts}
    monkeypatch.setattr(b, 'tokenize', lambda text: pytest.fail('Packed document must not be analyzed again'))
    for doc in packed.doc_texts:
        assert b.document_pairs(doc, packed) == expected[doc]
