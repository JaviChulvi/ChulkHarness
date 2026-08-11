# Software-engineer preset evaluation

This credential-free suite evaluates the public `SoftwareEngineer()` preset
through `Agent.run_result(...)`. A fixture creates a disposable Python project
with a broken `add()` function and one regression test. The scripted agent must
read the implementation, apply the smallest patch, run the test, and summarize
the verified result.

Run it directly or through the evaluation CLI:

```bash
python examples/software_engineer_eval/app.py
chulk eval run examples/software_engineer_eval/app.py:suite --json
```

The suite explicitly allows `apply_patch` and `run_cmd`; both operate only in
the fresh temporary workspace created for the case. Required graders check the
answer, run status, absence of errors, and exact tool sequence and arguments.
