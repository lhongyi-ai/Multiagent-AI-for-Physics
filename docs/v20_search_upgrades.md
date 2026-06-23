# V2 Search OS Upgrades

These upgrades strengthen the existing offline discovery runtime without changing the default permission model. They do not make live model calls and do not require network access.

## Optional Elo / Bradley-Terry Ranking

`SearchConfig.tournament_ranking_mode` defaults to `bounded`. Set it to `elo` or `bradley_terry` to write `elo_tournament_state.json`.

The rating path:

- updates ratings across deterministic candidate comparisons;
- preserves uncertainty;
- records completed pairings to avoid repeated comparisons;
- allocates bounded deeper comparison only to close or top-ranked pairings.

Relevant settings:

- `tournament_ranking_mode`
- `tournament_initial_rating`
- `tournament_k_factor`
- `tournament_initial_uncertainty`
- `tournament_close_match_gap`
- `tournament_max_deep_comparisons`

## Adaptive Compute Allocation

`adaptive_budget_allocation.json` records deterministic budget reallocation from historical yield:

- search strategy yield;
- candidate lineage yield;
- verifier-stage yield;
- model-role yield;
- duplicate penalties;
- falsification penalties;
- a preserved counterexample or contrarian branch floor.

The allocator remains bounded by `token_budget`, `model_call_budget`, and `verifier_call_budget`.

## Per-Role Model Routing

`provider_routing_plan.json` records provider-neutral routes for:

- generation;
- review;
- comparison;
- deep reasoning;
- novelty audit;
- meta-review.

In the current offline runner all routes resolve to deterministic mock mode unless a future live runner explicitly enables live model permission.

## Independent Reproduction

`reproduction_results.jsonl` checks top candidates through two independent local paths where feasible:

- direct stored verifier-score mean;
- independently recomputed check-list pass ratio.

Discrepancies are preserved in `reproduction_discrepancies.json` rather than averaged away.

## Workbench Views

The backend facade exposes artifact-backed views for:

- candidate archive;
- Elo/tournament state;
- strategy performance;
- adaptive allocation;
- verifier results;
- reproduction discrepancies;
- task queue;
- checkpoints;
- claim ledger;
- prediction ledger;
- expert feedback.

The frontend reads validated artifacts and does not duplicate scientific scoring logic.

## Offline Command

```bash
PYTHONPATH=src python -m coscientist.cli run-discovery \
  examples/discovery_search_fixture/project.yaml \
  --runs-dir runs \
  --run-id discovery-v20 \
  --force

PYTHONPATH=src python -m coscientist.cli validate-discovery runs/discovery-v20
```

To enable Elo on a project, add:

```yaml
search:
  tournament_ranking_mode: elo
  tournament_max_deep_comparisons: 2
```

## Limitations

These upgrades are process infrastructure. They do not prove scientific novelty, and they do not replace expert review, stronger verifiers, or domain-curated unresolved-problem dossiers.
