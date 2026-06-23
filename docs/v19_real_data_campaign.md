# V1.9 Real-Data Campaign Notes

V1.9 introduces an offline, source-aware campaign protocol for bounded real-data Atomic/AMO pilots.

## Real Versus Synthetic

V1.8 used synthetic hidden Hamiltonians. V1.9 uses curated local records copied into immutable run artifacts with source metadata, checksums, uncertainty fields, and train/validation/test visibility.

The included `87Rb` campaign is still a benchmark protocol, not a claim of new atomic physics.

## Source Curation

The curation pipeline:

1. reads local CSV snapshots;
2. preserves original values and original uncertainty;
3. normalizes supported frequency units to MHz;
4. assigns stable observation IDs;
5. records curation decisions;
6. detects duplicate/equivalent records;
7. writes dataset and split manifests;
8. copies source snapshots into the run directory;
9. validates checksums.

Conflicting or duplicate values are never silently averaged.

## Blind Evaluation

Agent-visible observations exclude test records. Evaluator-only data lives in `evaluator_only_reference.json` and is marked `hidden_from_agents`.

Validation checks that test observations are not present in `agent_visible_observations.jsonl`.

## Model Identifiability

Campaign runs write:

- `identifiability_results.jsonl`
- `equivalence_classes.json`
- `discriminating_observable_proposals.jsonl`

If models are equivalent on the current train split, the system preserves the group and proposes a discriminating observable rather than claiming a false unique winner.

## Model-Space Insufficiency

Campaign metrics include `model_space_insufficient`. The system is allowed to conclude that no current candidate family is adequate. The Rb-87 fixture currently selects a best-supported family within the tested candidate space.

## Adding A Second Real-Data Campaign

Create a directory like:

```text
examples/my_campaign/
  project.yaml
  sources/local_curated_observations.csv
```

The project should define:

- `campaign`
- `sources`
- `candidate_family_templates`
- `evaluator_only_reference`
- `open_problem_campaign_template`

Keep train/validation/test records separated by physical information, not random row selection.

## Graduating To An Open Problem

Before V2.0 open-problem use, fill `OpenProblemCampaignTemplate` with:

- a precise unresolved claim;
- current best baselines;
- candidate representation;
- automated verifier coverage;
- unresolved uncertainty;
- novelty definition;
- falsification criteria;
- expert review plan;
- allowed tools;
- stopping conditions;
- publication/disclosure policy.

Expert review remains mandatory.
