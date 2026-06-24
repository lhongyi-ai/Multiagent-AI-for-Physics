# V2.1 Superconductivity Domain Campaign Pack

V2.1 adds an offline, bounded superconductivity campaign for the mixed BCS / correlated-hopping question. It is a domain-specific software foundation, not a claim that the scientific problem is solved.

## What It Supports

- local superconductivity corpus records with claim-level curation status;
- fixture adapters for SuperCon-like, Materials Project-like, NOMAD-like, and OPTIMADE-like records;
- optional rebuildable SQLite artifact index;
- strict superconductivity model schemas;
- bounded phonon-only, correlated-hopping-only, mixed, and underdetermined model families;
- effective pairing-kernel construction;
- toy BCS gap/Tc calculation;
- free-energy and term-by-term energy decomposition;
- optical sum-rule proxy with cutoff warnings;
- doping/filling sweeps;
- material mapping with explicit missing fields;
- identifiability and equivalence classes;
- counterexample checks;
- differentiated scientific scores;
- frontend superconductivity workspace.

## Offline Commands

```bash
PYTHONPATH=src python -m coscientist.cli run-superconductivity-campaign \
  examples/superconductivity_bcs_campaign/project.yaml \
  --runs-dir runs \
  --run-id superconductivity-v21 \
  --force
```

```bash
PYTHONPATH=src python -m coscientist.cli validate-superconductivity-campaign \
  runs/superconductivity-v21
```

```bash
PYTHONPATH=src python -m coscientist.cli query-index \
  runs/superconductivity-v21 materials --limit 5
```

## SQLite Index

The SQLite database is optional and rebuildable:

```bash
PYTHONPATH=src python -m coscientist.cli rebuild-index runs/superconductivity-v21
PYTHONPATH=src python -m coscientist.cli validate-index runs/superconductivity-v21
```

The artifact filesystem remains the source of truth. The database stores paths, hashes, schema versions, and compact JSON payloads for indexing.

## Scientific Scope

The current solver is intentionally bounded:

- constant-DOS and toy lattice proxy only;
- separable effective pairing kernel only;
- scalar real gap proxy;
- no full Eliashberg solver;
- no DFT execution;
- no universal cuprate model;
- no automatic claim of discovery.

Valid outcomes include model-dependent energy separation, insufficient material mapping, non-identifiable channels, and expert-review-required.

## Frontend

Launch:

```bash
PYTHONPATH=src python -m coscientist.frontend
```

Open the Superconductivity tab and run:

```text
examples/superconductivity_bcs_campaign/project.yaml
```

The workbench displays scores, energy decomposition, optical warnings, material mappings, identifiability, and the campaign report.
