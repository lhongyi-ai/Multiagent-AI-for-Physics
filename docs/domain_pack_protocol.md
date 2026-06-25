# Domain Pack Protocol

Domain Packs isolate domain-specific science from generic runtime code.

A pack supplies:

- `domain_id` and `version`
- supported `ScientificTaskType` values
- search query templates
- source classification
- record normalization and validation
- readiness gates
- tool descriptors
- deterministic benchmark cases
- guardrails

The generic runtime must not hard-code LSCO, CrSe, XRD, or mathematical-physics rules. Those belong in `src/coscientist/domain_packs/`.

Current deterministic packs:

- `superconductivity_lsco`
- `magnetic_transport_crse`
- `mathematical_physics`
- `xrd_phase_identification`

Only fixture workflows are complete for the non-LSCO packs in this checkpoint.
