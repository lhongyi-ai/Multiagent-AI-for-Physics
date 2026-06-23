# Code Assistant Retrieval Fixture

This deterministic fixture exercises the V1 project workflow on an AI engineering topic rather than an urban climate topic. It is intentionally small and synthetic so default tests remain offline.

Run:

```bash
python -m coscientist.cli run-project research-projects/code_assistant_fixture/project.yaml --run-id code-assistant-pilot
```

Live-model smoke configuration, using the same fixture corpus and runtime provider selection:

```bash
python -m coscientist.cli run-project \
  research-projects/code_assistant_fixture/project.yaml \
  --provider openai \
  --live-model \
  --literature-mode fixture \
  --smoke \
  --run-id code-assistant-live-smoke
```

Mini live pilot configuration:

```bash
python -m coscientist.cli run-project \
  research-projects/code_assistant_fixture/project.yaml \
  --provider openai \
  --live-model \
  --literature-mode fixture \
  --max-model-calls 12 \
  --max-evolution-rounds 1 \
  --run-id code-assistant-live-mini
```
