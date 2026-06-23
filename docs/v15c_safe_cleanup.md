# V1.5C Safe Code Cleanup

This note records conservative cleanup performed during the V1.5C controlled-feedback evaluation work.

## Removed With High Confidence

### `src/coscientist/cli.py`: unused `load_fixture_corpus` import

- Evidence: `rg -n "load_fixture_corpus|compare-feedback|validate-feedback-ab|MetaReviewDecision|build_literature_artifacts|acquire_literature_only" src tests README.md RESEARCH_WORKFLOW.md examples research-projects`
- Result: `load_fixture_corpus` is used by `src/coscientist/literature/scholarly.py`, `tests/test_v15b_agents.py`, and `tests/test_v15c_feedback.py`, but no longer referenced by `src/coscientist/cli.py`.
- Compatibility: the public helper function remains in `coscientist.pilot.project_io`; only the unused CLI import was removed.
- Tests run after removal: `python -m pytest tests/test_schemas.py tests/test_v15b_agents.py` and `python -m pytest tests/test_v15c_feedback.py`.
- Replacement: none needed.
- Risk assessment: low. This removes an import only and does not change runtime behavior.

## Consolidated Safely

None. No helper functions or artifact readers were consolidated in this pass because the repository still benefits from explicit boundaries between V1 artifacts, V1.5B analysis, and V1.5C feedback experiments.

## Retained Because Usage Was Uncertain

- `MetaReviewDecision` in `src/coscientist/schemas/v15b.py`: retained because V1.5B artifacts and reports still use `meta_review_decisions_round_final.json`.
- `build_literature_artifacts` in `src/coscientist/pilot/runner.py`: retained because `acquire-literature`, `--dry-run`, and `--acquire-literature-only` still use it.
- Defensive live-permission gates in CLI and runner paths: retained even when tests do not exercise every negative branch.

## Retained For Backward Compatibility

- Existing V1 required artifacts and validators.
- Fixture, existing-corpus, and live literature modes.
- Mock provider and skipped live-provider test support.
- OpenAI-compatible provider interface and live-model permission gate.
- V1.5B proximity, grounding, and meta-review schemas/artifacts.

## Future Cleanup Candidates Requiring Human Review

- Shared helper extraction between `run_pilot_project` and `compare_feedback_project`: possible, but intentionally deferred to avoid hiding V1/V1.5C behavioral differences.
- A unified report builder for V1.5B and V1.5C: useful later, but current explicit branch reports are easier to audit.
- Mock provider call accounting: currently existing mock model usage reports zero call records. Changing that would affect existing tests and artifact expectations, so it should be considered separately.
