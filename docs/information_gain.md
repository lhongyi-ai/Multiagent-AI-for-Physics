# Information Gain

Expected Information Gain actions are deterministic priority records.

Priority currently uses:

```text
expected_information_gain * success_probability * feasibility / cost
```

The queue is written to `information_gain_queue.jsonl`. Live Agent Meeting receives the top actions and is instructed to choose repairs or verifiers from that queue.
