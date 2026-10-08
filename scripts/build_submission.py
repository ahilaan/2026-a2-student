"""Build the final source-only 2023CS10076.zip with repository-root paths.

Starter paths verified from 078249e765a38932f41914077814f20c2e11ddbf in
the original 2026-a2-student Git history. Dataset files stay local.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

STARTER_FILES = frozenset('''
.github/workflows/conformance.yml
.gitignore
README.md
assignment2.pdf
assignment2.tex
conftest.py
data/README.md
data/toy/candidates_dev.jsonl
data/toy/corpus.jsonl
data/toy/qrels_dev.txt
data/toy/queries_dev.tsv
docs/GRADING.md
docs/RELEVANCE_FEEDBACK_PRIMER.md
docs/SUBMISSION_INTERFACE.md
harness/__init__.py
harness/candidates_io.py
harness/leaderboard.py
harness/metrics.py
harness/noise_injection.py
harness/run_harness.py
harness/trec_io.py
requirements.txt
scripts/download_full_corpus.py
scripts/smoke_test.sh
submission/__init__.py
submission/adversarial_set.json
submission/corpus_utils.py
submission/feedback.py
submission/lm_utils.py
tests/test_interface_conformance.py
tests/test_leaderboard.py
tests/test_metrics.py
'''.split())
ADDITIONS = frozenset('''
docs/STARTER_README.md
scripts/build_submission.py
submission/analyzer.py
submission/prefix_lm.py
submission/bigram_lm.py
tests/test_feedback_models.py
tests/test_feedback_safety.py
tests/test_performance_equivalence.py
tests/test_prefix_lm.py
tests/test_bigram_lm.py
tests/test_shared_mu.py
tests/test_final_model.py
'''.split())
DATASET_FILES = frozenset(n for n in STARTER_FILES
                          if n.startswith('data/') and not n.endswith('.md'))
SOURCE_FILES = (STARTER_FILES - DATASET_FILES) | ADDITIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', default='2023CS10076', choices=['2023CS10076'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = sorted(SOURCE_FILES)
    missing = [name for name in expected if not (root / name).is_file()]
    if missing:
        raise RuntimeError('Missing required source files: ' + ', '.join(missing))
    for name in expected:
        path = Path(name)
        assert not set(path.parts) & {'.git', '.venv', '__pycache__', '.pytest_cache', 'runs'}
        assert path.suffix not in {'.pyc', '.zip', '.jsonl', '.tsv', '.csv', '.html', '.pkl', '.pickle'}
        assert path.suffix != '.json' or name == 'submission/adversarial_set.json'
    bonus = json.loads((root / 'submission/adversarial_set.json').read_text(encoding='utf-8'))
    assert isinstance(bonus, dict) and set(bonus) <= {'_comment'}, 'Bonus must remain a no-op'
    output_dir = root / 'output'
    output_dir.mkdir(exist_ok=True)
    output = output_dir / (args.entry + '.zip')
    canonical = root.parent / (args.entry + '.zip')
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix='.tmp', delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as archive:
            for name in [*expected, 'data/toy/']:
                info = zipfile.ZipInfo(name, (2026, 10, 8, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (0o40755 if name.endswith('/') else 0o100644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, b'' if name.endswith('/') else (root / name).read_bytes())
        with zipfile.ZipFile(temporary_path) as archive:
            assert archive.testzip() is None
            assert set(archive.namelist()) == SOURCE_FILES | {'data/toy/'}
            assert all(archive.read(name) == (root / name).read_bytes() for name in expected)
        temporary_path.replace(output)
        shutil.copy2(output, canonical)
        assert output.read_bytes() == canonical.read_bytes()
    finally:
        temporary_path.unlink(missing_ok=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print('Complete ZIP listing:')
    with zipfile.ZipFile(output) as archive:
        for name in archive.namelist():
            print(name)
    print('SHA256:', digest)
    print('Ready-to-submit ZIP:', canonical)
    print('Matching local copy:', output)


if __name__ == '__main__':
    main()
