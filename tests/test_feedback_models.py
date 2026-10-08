"""Independent mathematical and edge-case checks; shipped tests are unchanged."""
import json
import math
from collections import Counter
import pytest
from submission import feedback as f, lm_utils as lm

@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(f, 'PREFIX_LM_WEIGHT', 0.0)
    monkeypatch.setattr(lm, 'ANALYZER', 'raw')
    for key, value in dict(DIRICHLET_MU=1500.0, RM_DOC_MU=1000.0, RM3_LAMBDA_ORIG=0.65, FEEDBACK_TERMS=15, MIN_SUPPORT_FRAC=0.2).items():
        monkeypatch.setattr(f, key, value)
    docs = {'s1': 'query alpha alpha', 's2': 'query beta', 'ca': 'query alpha alpha alpha', 'cb': 'query beta beta beta', 'empty': ''}
    path = tmp_path / 'corpus.jsonl'
    path.write_text('\n'.join((json.dumps({'doc_id': d, 'text': t}) for d, t in docs.items())), encoding='utf-8')
    f.prepare(str(path))
    return f._STATS

def test_ql_matches_hand_formula_and_query_multiplicity(corpus, monkeypatch):
    monkeypatch.setattr(f, 'BIGRAM_LM_WEIGHT', 0.0)
    expected = 2 * math.log((1 + 1500 * 4 / 13) / (3 + 1500))
    assert f.score_candidates('query query', ['s1'], 1)[0][1] == pytest.approx(expected)

def test_rm1_includes_background_from_doc_where_term_absent(corpus):
    rm1, support = f._estimate_rm1({'s1': 0.25, 's2': 0.75}, corpus)
    expected = 0.25 * (2 + 1000 * 5 / 13) / 1003 + 0.75 * (1000 * 5 / 13) / 1002
    assert rm1['alpha'] == pytest.approx(expected)
    assert support['alpha'] == 1
    assert support['query'] == 2

def test_rm2_matches_direct_conditional_formula(corpus):

    def p(term, doc):
        tf = {'s1': {'query': 1, 'alpha': 2}, 's2': {'query': 1, 'beta': 1}}
        cf = {'query': 4, 'alpha': 5, 'beta': 4}
        length = {'s1': 3, 's2': 2}
        return (tf[doc].get(term, 0) + 1000 * cf[term] / 13) / (length[doc] + 1000)
    raw = {}
    for term in ('query', 'alpha', 'beta'):
        pw = [p(term, doc) for doc in ('s1', 's2')]
        conditional = sum((p('query', doc) * value for doc, value in zip(('s1', 's2'), pw))) / sum(pw)
        raw[term] = sum(pw) / 2 * conditional ** 2
    total = sum(raw.values())
    result = f._estimate_rm2(Counter({'query': 2}), ['s1', 's2'], corpus)
    assert result == pytest.approx({term: value / total for term, value in raw.items()})
    assert f._estimate_rm2(Counter({'query': 10000}), ['s1', 's2'], corpus)

def test_seed_posterior_normalized_and_soft(corpus):
    weights = f._seed_posteriors(Counter({'alpha': 1}), ['s1', 's2'], corpus)
    assert sum(weights.values()) == pytest.approx(1)
    assert weights['s1'] > weights['s2'] > 0

def test_support_filter_and_raw_probability_renormalization(corpus):
    query = Counter({'query': 1})
    rm1, support = f._estimate_rm1({'s1': 1.0}, corpus)
    one = f._feedback_evidence(query, ['s1'], corpus)
    total = rm1['query'] + rm1['alpha']
    assert one['selected'] == pytest.approx({'query': rm1['query'] / total, 'alpha': rm1['alpha'] / total})
    two = f._feedback_evidence(query, ['s1', 's2'], corpus)
    assert two['selected'] == {'query': 1.0}
    assert two['confidence'] == 0.0

def test_rm3_adaptive_mass_and_weighted_document_score(corpus):
    model = f._build_rm3_model(Counter({'query': 2}), {'alpha': 1.0})
    expansion = 0.35 / 15
    assert model == pytest.approx({'query': 1 - expansion, 'alpha': expansion})
    assert sum(model.values()) == pytest.approx(1)
    expected = (1 - expansion) * math.log((1 + 1500 * 4 / 13) / 1503) + expansion * math.log((2 + 1500 * 5 / 13) / 1503)
    assert f._rank_with_term_distribution(model, ['s1'], 1, corpus)[0][1] == pytest.approx(expected)

def test_actual_seed_changes_ranking_and_may_be_outside_candidates(corpus):
    candidates = ['ca', 'cb']
    a = f.relevance_model_feedback('query', ['s1'], candidates)
    b = f.relevance_model_feedback('query', ['s2'], candidates)
    assert a[0][0] == 'ca'
    assert b[0][0] == 'cb'
    assert {doc for doc, _ in a + b} <= set(candidates)
    f.score_candidates('query', candidates)
    assert f.relevance_model_feedback('query', ['s1'], candidates) == a

@pytest.mark.parametrize('query', ['', 'zzzoov', 'query', 'query zzzoov'])
def test_fallback_edges_and_finite_unique_sorted_results(corpus, query):
    candidates = ['empty', 'ca', 'cb', 'ca']
    base = f.score_candidates(query, candidates, 100)
    for seed in ([], ['missing', 'empty']):
        assert f.relevance_model_feedback(query, seed, candidates, 100) == base
    for result in (base, f.relevance_model_feedback(query, ['s1'], candidates, 100)):
        assert len({doc for doc, _ in result}) == len(result)
        assert all((math.isfinite(score) for _, score in result))
        assert [score for _, score in result] == sorted((score for _, score in result), reverse=True)
    assert f.score_candidates(query, candidates, 0) == []
    assert f.score_candidates(query, candidates, -1) == []
    assert f.relevance_model_feedback(query, ['s1'], candidates, 0) == []

def test_oov_ties_keep_candidate_order_and_seed_duplicates_are_ignored(corpus):
    assert [doc for doc, _ in f.score_candidates('zzzoov', ['cb', 'ca', 'cb'])] == ['cb', 'ca']
    assert f.relevance_model_feedback('query', ['s1', 's1'], ['ca', 'cb']) == f.relevance_model_feedback('query', ['s1'], ['ca', 'cb'])
