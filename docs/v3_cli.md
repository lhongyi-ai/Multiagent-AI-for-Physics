# V3 CLI

Run the deterministic proof-carrying demo:

```bash
PYTHONPATH=src python -m coscientist.cli run-v3-proof-search-demo \
  --runs-dir runs \
  --run-id v3-proof-search-demo \
  --force
```

Validate the artifacts:

```bash
PYTHONPATH=src python -m coscientist.cli validate-v3-proof-search runs/v3-proof-search-demo
```

The demo writes:

- `compiled_problem.json`
- `proof_obligations.jsonl`
- `scientific_state_snapshots.jsonl`
- `tool_capability_manifest.json`
- `tool_gaps.jsonl`
- `tool_build_records.jsonl`
- `candidate_archive.jsonl`
- `action_executions.jsonl`
- `verifier_results.jsonl`
- `independent_verification_results.jsonl`
- `certificates.jsonl`
- `claim_transition_records.jsonl`
- `state_disputes.jsonl`
- `final_adjudication.json`
- `run_summary.md`

The command is offline and deterministic.
