# Problem Definition

## Context

Heart attacks (myocardial infarctions) are one of the leading causes of death worldwide. Early identification of risk factors can help healthcare providers take preventive measures and improve patient outcomes.

This project applies **Bayesian Networks** to model the probabilistic relationships between lifestyle factors, physiological conditions, and heart attack risk. Unlike traditional machine learning approaches that require large datasets, Bayesian Networks can incorporate expert knowledge through defined conditional probability distributions.

## Problem Statement

Given a set of observable risk factors:
- **Smoking status** (smoker or non-smoker)
- **Exercise habits** (regular exercise or sedentary)
- **Blood pressure** (high or normal)
- **Cholesterol levels** (high or normal)

**Objective**: Predict the probability of suffering a heart attack before age 70.

## Approach

We use a **directed acyclic graph (DAG)** to represent causal relationships between variables:

```
    Smoking ─────┐
                 ├──→ Blood Pressure ──→ Heart Attack
    Exercise ────┘          │
        │                   │
        └──→ Cholesterol ───┘
```

### Variables

| Variable | Symbol | States | Description |
|----------|--------|--------|-------------|
| Smoker | F | V (Yes), F (No) | Whether the patient smokes |
| Exercise | E | V (Yes), F (No) | Regular exercise habits |
| High Pressure | P | A (Yes), B (No) | Blood pressure condition |
| High Cholesterol | C | A (Yes), B (No) | Cholesterol level |
| Heart Attack | A | A (Yes), B (No) | Heart attack occurrence |

## Inference Method

We apply the **Variable Elimination** algorithm for exact probabilistic inference. This method:

1. Represents probability tables as **factors**
2. Performs factor multiplication (similar to JOIN in databases)
3. Marginalizes over hidden variables (summation)
4. Returns exact conditional probabilities

The ordering of eliminations affects computational complexity. Finding the optimal order is NP-HARD, so we use heuristics like min-neighbors, min-weight, or min-fill.

## Success Criteria

1. Implement a working Bayesian Network model
2. Answer probabilistic queries about heart attack risk
3. Validate manual calculations against library implementation (pgmpy)
4. Provide an API for real-time predictions
