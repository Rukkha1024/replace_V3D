# Environment notes (sandbox)

- This sandbox container **does not have `conda` installed**, so local validation was executed with the system `python` interpreter.
- `polars` is **not installed** in the sandbox Python environment. The new ankle-torque pipeline uses **polars when available**, but includes a **pandas fallback** so it can still run without polars.

On your lab machine, please run via the repo rule:

```bash
conda run -n module python ...
```
