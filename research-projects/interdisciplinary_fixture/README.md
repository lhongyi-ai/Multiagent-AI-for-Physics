# Deterministic Pilot Project: Urban Heat-Island Mitigation

This directory is a small V1 pilot fixture for the grounded research and evaluation loop.

The project asks which safe, testable mechanisms could explain why different urban heat-island mitigation strategies work differently across neighborhoods.

The included literature corpus is a deterministic test corpus, not a complete literature review. It is designed to contain complementary and mildly conflicting evidence so the workflow can exercise evidence links, citation verification, ranking, evolution, comparison, and human-review artifacts without live network access.

## Run

```bash
python -m coscientist.cli run-project research-projects/interdisciplinary_fixture/project.yaml --run-id urban-heat-pilot
```

Then inspect:

```bash
python -m coscientist.cli validate-artifacts runs/urban-heat-pilot
python -m coscientist.cli compare-rounds runs/urban-heat-pilot
python -m coscientist.cli build-review-package runs/urban-heat-pilot
```

## Files

- `project.yaml`: persistent research-project specification.
- `corpus.jsonl`: small offline fixture literature corpus.
- `rubric.yaml`: pilot evaluation dimensions.
- `human_review_template.md`: reviewer decision template.
- `expected_artifacts.md`: expected completed-run artifacts.

