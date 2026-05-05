# Heart Attack Risk — Bayesian Network with Expert CPDs

A small Bayesian network for heart-attack risk where the conditional
probability tables come from clinical priors, not learned from data. The
graph encodes the four classic risk-factor relationships (smoking,
exercise, blood pressure, cholesterol) and the inference is exact — no
Monte Carlo, no approximation.

## What I built

- **Hand-specified DAG with expert CPDs** (`src/model.py`): five binary
  variables, four parent–child links, all conditional tables hardcoded
  from medical literature so the model is interpretable end to end.
- **Variable Elimination from scratch**, then validated against pgmpy's
  built-in inference engine — both produce the same posteriors.
- **FastAPI service** with `/predict`, `/query`, `/variables` endpoints,
  rate-limited and CORS-enabled via `src/api_common.py`.
- **Marginalization at request time**: the client provides only the risk
  factors they know; the rest are summed out automatically.

## Why it matters

When you don't have a labeled dataset (and most clinical settings don't),
expert-CPD Bayesian networks are still useful — and they're auditable in a
way a neural net never will be. Every probability the model emits is
traceable back to one CPD entry.

## Tech stack

pgmpy (Variable Elimination, DiscreteBayesianNetwork) · FastAPI · NumPy · pandas

## Quickstart

```bash
# install (one-time)
pip install -e ".[api,notebook,dev]"

# run the API
uvicorn api.app:app --reload --port 8006

# predict heart-attack probability for a smoker with high cholesterol
curl -X POST http://localhost:8006/predict \
  -H 'Content-Type: application/json' \
  -d '{"smoker": true, "high_cholesterol": true}'

# raw query: P(A | F=V) using the variable-letter convention
curl -X POST http://localhost:8006/query \
  -H 'Content-Type: application/json' \
  -d '{"target": "A", "evidence": {"F": "V"}}'
```

## Live demo

Hosted on Hugging Face Spaces:
[kevinreyesds/heartattack-bn](https://huggingface.co/spaces/kevinreyesds/heartattack-bn)
*(wired through the portfolio at `/projects/heartattack/demo`)*

## Tests

```bash
pytest
ruff check .
```

Tests cover the network structure, exact-inference outputs, the `/predict`
and `/query` API contracts, and a few sanity checks against pgmpy's
reference implementation.

## Repository

[Portfolio-KRV/heartattack](https://github.com/Portfolio-KRV/heartattack)

## License

[MIT](LICENSE)
