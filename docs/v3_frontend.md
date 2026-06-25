# V3 Frontend

The Gradio workbench includes a focused `V3 Proof Search` tab.

It exposes backend artifacts for:

- scientific state snapshots;
- proof obligations;
- proof-carrying claim transitions;
- certificates;
- tool gaps;
- tool build records;
- candidate archive;
- state disputes;
- independent verification;
- final adjudication.

The tab calls the same backend services as the CLI:

- `OfflineDiscoveryFrontend.run_v3_proof_search_demo()`
- `OfflineDiscoveryFrontend.validate_v3_proof_search()`

For superconductivity decomposition/gauge/observability questions, Live Agent Room round zero also runs the V3 proof-search context and writes it into `meeting_tool_context.json`. The transcript shows:

```text
Tool Call: v3_proof_carrying_scientific_search
```

This is intended to prevent agents from repeating "derive the Hamiltonian later" when a bounded proof-carrying artifact already exists.
