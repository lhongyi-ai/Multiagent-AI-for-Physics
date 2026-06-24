# V2.2 Superconductivity Theory-Discrimination Campaign

V2.2 adds a bounded superconductivity campaign workspace for microscopic theory discrimination. It keeps the artifact filesystem as the source of truth and remains deterministic by default.

The central question is whether phonon-dominated, Hirsch-style correlated-hopping, mixed, and alternate kinetic mechanisms can be distinguished with jointly constrained doping-resolved observables from one material family.

## Offline Fixture Run

```bash
PYTHONPATH=src python -m coscientist.cli run-v22-campaign \
  examples/v22_superconductivity_real_data/project.yaml \
  --runs-dir runs \
  --run-id v22-fixture \
  --force

PYTHONPATH=src python -m coscientist.cli validate-v22-campaign runs/v22-fixture
```

This writes candidate models, microscopic Hamiltonians, derivations, fingerprints, adversarial tests, reproduction records, a concrete experiment proposal, a claim ledger, an expert-review package, and a SQLite artifact index.

## Live Data Smoke

Live network access is never inferred. Use:

```bash
PYTHONPATH=src python -m coscientist.cli test-data-connections --live-network \
  --runs-dir runs \
  --run-id v22-data-live \
  --force
```

Current bounded public smoke checks cover OpenAlex, Crossref, arXiv, DataCite, Zenodo, NOMAD, and OPTIMADE when reachable. Unpaywall requires `UNPAYWALL_EMAIL`. Materials Project requires `MATERIALS_PROJECT_API_KEY`. Providers without a stable unauthenticated JSON smoke endpoint are reported as `unavailable`, not `connected`.

## Live Model Smoke

Live model access is also explicit:

```bash
PYTHONPATH=src python -m coscientist.cli test-live-models --live-model \
  --runs-dir runs \
  --run-id v22-model-live \
  --force
```

OpenRouter routes use `OPENROUTER_API_KEY` first, then `OPENAI_API_KEY`; the default base URL is `https://openrouter.ai/api/v1`. You can set per-role models with `GENERATOR_MODEL`, `REVIEWER_MODEL`, `ADVERSARIAL_MODEL`, `EVOLUTION_MODEL`, `RANKER_MODEL`, and `META_REVIEW_MODEL`.

## Full Bounded Campaign With Live Data Status

```bash
PYTHONPATH=src python -m coscientist.cli run-v22-campaign \
  examples/v22_superconductivity_real_data/project.yaml \
  --live-network \
  --runs-dir runs \
  --run-id v22-live-network \
  --force
```

This records live database smoke status and one-record response snapshots where public endpoints connect. It still uses deterministic mock model reasoning unless `--live-model` is also supplied and credentials are configured.

## Important Limits

V2.2 does not claim a superconductivity discovery. It can produce bounded constraints, equivalence classes, adversarial objections, held-out prediction artifacts, and experiment proposals. Public scientific claims remain blocked behind expert review.
