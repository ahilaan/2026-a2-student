"""Safety additions; no changes to the course's grading tests."""
import json
import math
from collections import Counter
import pytest
from submission import feedback as f, lm_utils as lm
from submission.analyzer import analyze, porter_stem

@pytest.mark.parametrize('word,expected', [('caresses', 'caress'), ('ponies', 'poni'), ('ties', 'ti'), ('cats', 'cat'), ('agreed', 'agre'), ('plastered', 'plaster'), ('motoring', 'motor'), ('hopping', 'hop'), ('filing', 'file'), ('relational', 'relat'), ('conditional', 'condit'), ('vietnamization', 'vietnam'), ('triplicate', 'triplic'), ('revival', 'reviv'), ('probate', 'probat'), ('controll', 'control'), ('sars2', 'sars2')])
def test_porter_known_examples(word, expected):
    assert porter_stem(word) == expected

def test_analyzer_modes_and_hyphenated_tokens():
    assert analyze('The vaccines and immunity', 'raw') == ['the', 'vaccines', 'and', 'immunity']
    assert analyze('The vaccines and immunity', 'stop') == ['vaccines', 'immunity']
    assert analyze('The vaccines and immunity', 'stop_porter') == ['vaccin', 'immun']
    assert analyze('COVID-19 / SARS-CoV-2', 'stop_porter') == ['covid', '19', 'sar', 'cov', '2']

@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(lm, 'ANALYZER', 'raw')
    for key, value in dict(CONFIDENCE_FLOOR=0.05, ANCHOR_EXPONENT=1.0, FEEDBACK_TERMS=15, RM3_LAMBDA_ORIG=0.85, DIRICHLET_MU=1500.0).items():
        monkeypatch.setattr(f, key, value)
    docs = {'a': 'query alpha alpha', 'b': 'query beta beta', 'noise': 'unrelated poison poison', 'ca': 'query alpha alpha alpha', 'cb': 'query beta beta beta'}
    path = tmp_path / 'corpus.jsonl'
    path.write_text('\n'.join((json.dumps({'doc_id': d, 'text': t}) for d, t in docs.items())), encoding='utf-8')
    f.prepare(str(path))
    return f._STATS

def test_query_terms_keep_rm1_mass_and_distribution_normalizes(prepared):
    evidence = f.feedback_diagnostics('query', ['a'])
    assert evidence['selected']['query'] > 0
    assert evidence['selected']['alpha'] > 0
    assert sum(evidence['selected'].values()) == pytest.approx(1)
    assert sum(evidence['model'].values()) == pytest.approx(1)
    assert 0.85 <= evidence['effective_lambda'] < 1
    assert evidence['confidence'] > 0

def test_coherent_but_offquery_seed_falls_back_exactly(prepared):
    pool = ['ca', 'cb']
    evidence = f.feedback_diagnostics('query', ['noise'])
    assert evidence['confidence'] == 0
    assert f.relevance_model_feedback('query', ['noise'], pool) == f.score_candidates('query', pool)

def test_anchor_coverage_and_ess_are_seed_only_and_finite(prepared):
    evidence = f.feedback_diagnostics('query', ['a', 'noise'])
    assert all((math.isfinite(evidence[key]) for key in ['confidence', 'posterior_entropy', 'posterior_ess']))
    assert 1 <= evidence['posterior_ess'] <= 2
    assert evidence['coverage_lower_quartile'] == 0
    alpha = next((term for term in evidence['terms'] if term['term'] == 'alpha')) if any((term['term'] == 'alpha' for term in evidence['terms'])) else None
    if alpha:
        assert alpha['anchored_coverage'] == 1
    before = f.feedback_diagnostics('query', ['a'])
    f.score_candidates('query', ['ca', 'cb'])
    assert f.feedback_diagnostics('query', ['a']) == before

def test_same_scale_score_fusion_is_distribution_interpolation(prepared):
    original = {'query': 1.0}
    model = {'query': 0.8, 'alpha': 0.2}
    fused = {'query': 0.9, 'alpha': 0.1}
    for doc in ('ca', 'cb'):
        expected = 0.5 * f._doc_ql_score(doc, original, prepared) + 0.5 * f._doc_ql_score(doc, model, prepared)
        assert f._doc_ql_score(doc, fused, prepared) == pytest.approx(expected)

def test_candidate_mu_is_used_by_shared_model(prepared, monkeypatch):
    a = f.score_candidates('query alpha', ['ca'])[0][1]
    monkeypatch.setattr(f, 'DIRICHLET_MU', 100.0)
    b = f.score_candidates('query alpha', ['ca'])[0][1]
    assert a != b

def test_sharp_evidence_remains_seed_sensitive(prepared):
    pool = ['ca', 'cb']
    assert f.relevance_model_feedback('query', ['a'], pool)[0][0] == 'ca'
    assert f.relevance_model_feedback('query', ['b'], pool)[0][0] == 'cb'

def test_shipped_defaults_remain_seed_sensitive(tmp_path):
    path = tmp_path / 'actual_defaults.jsonl'
    docs = {'a': 'kinase alpha alpha', 'b': 'kinase beta beta', 'ca': 'kinase alpha alpha alpha', 'cb': 'kinase beta beta beta'}
    path.write_text('\n'.join((json.dumps({'doc_id': d, 'text': t}) for d, t in docs.items())), encoding='utf-8')
    f.prepare(str(path))
    pool = ['ca', 'cb']
    assert f.relevance_model_feedback('kinase', ['a'], pool)[0][0] == 'ca'
    assert f.relevance_model_feedback('kinase', ['b'], pool)[0][0] == 'cb'
