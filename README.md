# COL764 Assignment 2 — 2023CS10076

The same query-likelihood candidate scorer is used in Tracks A, B and C.
Queries and documents use stopword removal followed by the student's Porter
stemmer. Ranking is restricted to the supplied candidate pool, with duplicate
IDs removed and candidate order retained on exact ties.

## Final model

- Full-document unigram Dirichlet LM: `DIRICHLET_MU = 500`.
- Word probabilities mix 0.75 full-document LM with 0.25 normalized LM of the
  first 32 analyzed tokens, smoothed with `PREFIX_LM_MU = 100`.
- Ordered adjacent-pair Dirichlet LM: weight 0.10, mu 500, with the normalized
  product of collection unigram probabilities as its prior. Word and pair
  log-likelihoods use the existing query-mass normalization.
- Supplied feedback seeds drive RM1 and RM3: document mu 250, original-query
  weight 0.88, 15 feedback terms and minimum support fraction 0.20. Coverage,
  query anchoring, posterior weighting and confidence gating determine the
  effective feedback mass. Original query pairs remain fixed.
- Setting `RM3_LAMBDA_ORIG = 1.0` returns exactly the base scorer's document IDs
  and floating-point scores. Pure unigram QL, RM2 and feedback diagnostics
  remain independently callable for mathematical verification.

Preparation and retrieval run in one process. Compact document statistics and
bounded document-content caches avoid repeated analysis. No adversarial bonus
is attempted; the starter bonus file contains a comment only.

## Validation and packaging

```bash
pip install -r requirements.txt
python -m pytest tests -q
python scripts/build_submission.py --entry 2023CS10076
```

The builder writes `../2023CS10076.zip` and an identical
`output/2023CS10076.zip`, preserving repository-root paths. Dataset files are
excluded as required. The course's toy fixtures must be supplied locally for
the toy conformance tests; they are retained in the working repository.

Original course documentation is preserved in `assignment2.pdf`,
`assignment2.tex`, `docs/`, and [the starter README](docs/STARTER_README.md).
The source-only file set is derived from starter Git commit
`078249e765a38932f41914077814f20c2e11ddbf`, plus the active implementation,
focused regression tests and submission builder.

AI-use disclosure: OpenAI Codex assisted with implementation, cleanup and
validation. Provenance includes the supplied course starter/harness, the
student's unigram/RM1/RM2/RM3 code and the student's Assignment 1 Porter
implementation. The positional and ordered-pair LM arithmetic was implemented
locally, without an external retrieval library.
