# Remote Acquisition Worker Extension

The acquisition layer uses an `AcquisitionExecutor` protocol so the same task can run in-process or later on a remote worker.

Current implementations:

- `MockAcquisitionExecutor`: deterministic tests.
- `LocalAcquisitionExecutor`: local fixture/existing/live execution.

Future implementation:

- `RemoteAcquisitionExecutor`: submit an `AcquisitionTask` to a cloud VM, container, or persistent worker; poll status; retrieve `AcquisitionResult` artifacts.

Remote workers must preserve the same permission, artifact, staging, promotion, and provenance semantics as the local executor. They must not bypass live-network permission gates or write directly to canonical data.
