# Tool Capability Registry

V3.0 makes missing scientific capabilities explicit.

## Registry

`CapabilityAwareToolRegistry` stores tool descriptors with:

- tool ID;
- version;
- declared capabilities;
- registration status;
- validation certificate ID;
- deterministic/network flags.

Only `registered` or `validated` tools can be added as executable tools. Generated tools require a validation certificate before registration.

## Tool Gaps

`detect_tool_gaps()` compares proof-obligation requirements against registered capabilities and emits `ToolGap` records.

`build_missing_tool_actions()` converts those gaps into `BUILD_MISSING_TOOL` actions.

## Current Demo

The V3 deterministic demo first records missing capabilities, then resolves them with audited deterministic tool-build records and source-controlled bundled fixtures. This is a conservative local simulation of the larger future tool-synthesis pipeline; it does not run arbitrary generated code.
