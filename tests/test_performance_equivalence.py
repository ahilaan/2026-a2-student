"""Verify removed work without wall-clock gates or weakened semantic tests."""
from collections import Counter
import sys
import subprocess
import pytest
from submission import feedback as f, lm_utils as lm

@pytest.fixture
def prepared(monkeypatch):
    monkeypatch.setattr(lm, 'ANALYZER', 'raw')
    stats = lm.CollectionStats.from_corpus([('a', 'alpha beta gamma'), ('b', 'alpha beta delta')])
    monkeypatch.setattr(f, '_STATS', stats)
    monkeypatch.setattr(f, '_DOC_FREQ', stats.doc_freq)
    monkeypatch.setattr(f, '_NUM_DOCS', 2)
    return stats

def test_production_evidence_omits_only_diagnostics(prepared, monkeypatch):
    original = f._doc_ql_score
    calls = []

    def score(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(f, '_doc_ql_score', score)
    query = Counter({'alpha': 1})
    minimal = f._feedback_evidence(query, ['a', 'b'], prepared, diagnostics=False)
    assert calls == ['a', 'b']
    calls.clear()
    full = f._feedback_evidence(query, ['a', 'b'], prepared)
    assert calls == ['a', 'b', 'a', 'b']
    assert minimal == {key: full[key] for key in ['model', 'confidence']}

@pytest.mark.parametrize('query,expected_calls', [(Counter({'alpha': 1}), 1), (Counter({'the': 1, 'alpha': 1}), 2)])
def test_coverage_prepared_once_per_distinct_query(prepared, monkeypatch, query, expected_calls):
    original = f._prepare_coverage
    calls = []

    def prepare(*args):
        calls.append(args[0])
        return original(*args)
    monkeypatch.setattr(f, '_prepare_coverage', prepare)
    f._feedback_evidence(query, ['a', 'b'], prepared)
    assert len(calls) == expected_calls

def test_rank_uses_identical_scalar_operations(prepared, monkeypatch):
    terms = Counter({'alpha': 2, 'unknown': 1, 'beta': 3})
    ids = ['b', 'a', 'b']
    expected = sorted([(d, f._doc_ql_score(d, terms, prepared)) for d in dict.fromkeys(ids)], key=lambda pair: pair[1], reverse=True)
    assert f._rank_with_term_distribution(terms, ids, 10, prepared) == expected

def test_import_and_zero_weight_prepare_are_serial_and_skip_lexical(tmp_path):
    corpus = tmp_path / 'corpus.jsonl'
    corpus.write_text('{"doc_id":"a","text":"alpha beta"}\n')
    script = "\nimport multiprocessing, sys\nfrom submission import feedback as f\nassert not multiprocessing.active_children()\nf.prepare(sys.argv[1])\nassert not multiprocessing.active_children()\nassert 'submission.lexical_utils' not in sys.modules\nassert 'submission.f2exp_auxiliary' not in sys.modules\nassert 'experiments.prepare_parallel' not in sys.modules\nf.score_candidates('alpha',['a'])\nf.relevance_model_feedback('alpha',['a'],['a'])\nassert not multiprocessing.active_children()\n"
    subprocess.run([sys.executable, '-c', script, str(corpus)], check=True)
