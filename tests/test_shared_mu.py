"""One QL retriever across all tracks; disabling feedback preserves scores."""
import math
import pytest
from submission import feedback as f, lm_utils as lm

@pytest.fixture
def corpus(monkeypatch):
    monkeypatch.setattr(f, 'PREFIX_LM_WEIGHT', 0.0)
    monkeypatch.setattr(f, 'BIGRAM_LM_WEIGHT', 0.0)
    monkeypatch.setattr(lm, 'ANALYZER', 'raw')
    stats = lm.CollectionStats.from_corpus([('a', 'alpha alpha beta'), ('b', 'alpha gamma gamma gamma'), ('tie', 'alpha alpha beta'), ('empty', '')])
    monkeypatch.setattr(f, '_STATS', stats)
    monkeypatch.setattr(f, '_DOC_FREQ', stats.doc_freq)
    monkeypatch.setattr(f, '_NUM_DOCS', len(stats.doc_texts))
    return stats

def test_mu75_matches_scalar_query_likelihood(corpus, monkeypatch):
    monkeypatch.setattr(f, 'DIRICHLET_MU', 75.0)
    terms = {'alpha': 2, 'beta': 1, 'unseen': 1}
    pool = ['tie', 'a', 'empty', 'b', 'tie']

    def scalar(doc):
        counts = corpus.doc_term_counts(doc)
        length = corpus.doc_lengths[doc]
        return sum((weight * (-50.0 if corpus.collection_prob(term) <= 0 else math.log((counts.get(term, 0) + 75 * corpus.collection_prob(term)) / (length + 75))) for term, weight in terms.items()))
    expected = sorted([(doc, scalar(doc)) for doc in dict.fromkeys(pool)], key=lambda row: row[1], reverse=True)
    assert f.score_candidates('alpha alpha beta unseen', pool, 10) == expected
    assert f.unigram_score_candidates('alpha alpha beta unseen', pool, 10) == expected

@pytest.mark.parametrize('mu', [75.0, 150.0, 500.0, 3000.0])
@pytest.mark.parametrize('query', ['alpha beta', 'alpha alpha beta', 'alpha unseen', 'unseen', ''])
def test_disabled_feedback_matches_track_a_rankings_and_float_bits(corpus, monkeypatch, mu, query):
    monkeypatch.setattr(f, 'DIRICHLET_MU', mu)
    monkeypatch.setattr(f, 'RM3_LAMBDA_ORIG', 1.0)
    pool = ['tie', 'a', 'empty', 'b', 'tie']
    expected = f.score_candidates(query, pool, 10)
    actual = f.relevance_model_feedback(query, ['a', 'b'], pool, 10)
    assert [(doc, score.hex()) for doc, score in actual] == [(doc, score.hex()) for doc, score in expected]

def test_empty_seed_uses_exact_same_mu75_retriever(corpus, monkeypatch):
    monkeypatch.setattr(f, 'DIRICHLET_MU', 75.0)
    query, pool = ('alpha beta', ['tie', 'a', 'empty', 'b', 'tie'])
    assert f.relevance_model_feedback(query, [], pool, 10) == f.score_candidates(query, pool, 10)
