"""Frozen parameters and exact single-model conformance for the final system."""
import json
import math
from pathlib import Path

import pytest
from submission import feedback as f, lm_utils as lm


def test_final_parameters():
    expected = dict(DIRICHLET_MU=500., RM_DOC_MU=250., RM3_LAMBDA_ORIG=.88,
                    PREFIX_LM_WEIGHT=.25, PREFIX_LM_TOKENS=32, PREFIX_LM_MU=100.,
                    BIGRAM_LM_WEIGHT=.10, BIGRAM_LM_MU=500., FEEDBACK_TERMS=15,
                    MIN_SUPPORT_FRAC=.20, CONFIDENCE_FLOOR=.4,
                    SEED_POSTERIOR_TEMPERATURE=1., QUERY_COVERAGE_FLOOR=.25,
                    ANCHOR_EXPONENT=1.)
    assert {name: getattr(f, name) for name in expected} == expected
    assert lm.ANALYZER == 'stop_porter'


@pytest.mark.parametrize('query', ['', 'unseenxyz', 'kinase', 'kinase alpha',
                                   'alpha alpha beta', 'beta alpha', 'alpha unseenxyz'])
def test_feedback_off_exact_ids_float_bits_orders_and_seeds(tmp_path, monkeypatch, query):
    docs = {'a': 'kinase alpha alpha beta ' + 'gamma ' * 40,
            'tie': 'kinase alpha alpha beta ' + 'gamma ' * 40,
            'b': 'kinase beta beta alpha ' + 'delta ' * 40,
            'outside': 'kinase alpha beta alpha gamma', 'empty': ''}
    path = tmp_path / 'corpus.jsonl'
    path.write_text('\n'.join(json.dumps(dict(doc_id=d, text=t)) for d, t in docs.items()),
                    encoding='utf-8')
    f.prepare(str(path))
    monkeypatch.setattr(f, 'RM3_LAMBDA_ORIG', 1.)
    for pool in [['tie', 'b', 'a', 'empty', 'a'], ['a', 'tie', 'empty', 'b'], []]:
        for seed in [[], ['missing', 'empty'], ['outside'], ['a', 'b'],
                     ['outside', 'outside', 'b']]:
            # The first feedback call follows prepare directly, with no base call.
            actual = f.relevance_model_feedback(query, seed, pool, 100)
            expected = f.score_candidates(query, pool, 100)
            assert [(d, s.hex()) for d, s in actual] == [(d, s.hex()) for d, s in expected]
            ids = [d for d, _ in actual]
            assert len(ids) == len(set(ids))
            assert set(ids) <= set(pool)


def test_unigram_dirichlet_mu500_hand_example(monkeypatch):
    monkeypatch.setattr(lm, 'ANALYZER', 'raw')
    stats = lm.CollectionStats.from_corpus([('a', 'alpha alpha beta'), ('b', 'beta gamma gamma')])
    monkeypatch.setattr(f, '_STATS', stats)
    a = 2 * math.log((2 + 500 * (2 / 6)) / 503) + math.log((1 + 500 * (2 / 6)) / 503) - 50.
    b = 2 * math.log((500 * (2 / 6)) / 503) + math.log((1 + 500 * (2 / 6)) / 503) - 50.
    actual = f.unigram_score_candidates('alpha alpha beta unseen', ['b', 'a', 'a'])
    assert [(d, s.hex()) for d, s in actual] == [('a', a.hex()), ('b', b.hex())]


def test_bonus_is_noop():
    path = Path(f.__file__).with_name('adversarial_set.json')
    assert set(json.loads(path.read_text(encoding='utf-8'))) <= {'_comment'}
