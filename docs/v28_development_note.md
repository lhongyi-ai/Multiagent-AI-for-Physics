# V2.8 Development Note

Inspected files before implementation:

- `src/coscientist/core/optimizer_v2.py`
- `src/coscientist/core/tasks.py`
- `src/coscientist/core/domain_packs.py`
- `src/coscientist/domain_packs/superconductivity_lsco.py`
- `src/coscientist/live_agents.py`
- `src/coscientist/claim_dag.py`
- `src/coscientist/schemas/v23.py`
- `src/coscientist/frontend.py`
- `src/coscientist/superconductivity/minimal_model.py`
- `src/coscientist/superconductivity/phase2_data.py`
- `src/coscientist/superconductivity/energy_decomposition.py`
- `tests/test_v23_live_agent_workbench.py`
- `tests/test_v26_domain_packs_optimizer.py`
- `tests/test_v27_energy_decomposition_audit.py`

Baseline before V2.8 edits:

- `PYTHONPATH=src pytest -q`
- Result: `177 passed, 4 skipped`

Implementation plan:

1. Add a domain-independent scientific action execution layer that can adapt existing Optimizer V2 queue rows into executable action requests.
2. Register existing deterministic tools instead of creating duplicate scientific logic.
3. Persist one action bundle per selected action with policy, invocation, result, verifier, Claim DAG diff, optimizer diff, and summary artifacts.
4. Update the Claim DAG artifact set transactionally enough for the current local JSON/SQLite layer: create seed DAG artifacts when absent, append action-backed checks/blockers/contradictions, then rebuild the SQLite index.
5. Integrate exactly one eligible action execution at the start of each productive Live Agent Meeting round, then inject compact action state into the next prompts and frontend tables.
6. Preserve guardrails: no default live calls, no silent mock fallback in live mode, no unique phonon-versus-kinetic mechanism percentages, and no LSCO Tier A/B promotion to Tier C.
