# Methodology

## Bayesian Network Construction

### 1. Model Definition

We use the `pgmpy` library to define a Bayesian Network with the following structure:

```python
BayesianModel([
    ("F", "P"),  # Smoker -> Pressure
    ("E", "P"),  # Exercise -> Pressure
    ("E", "C"),  # Exercise -> Cholesterol
    ("P", "A")   # Pressure -> Heart Attack
])
```

### 2. Conditional Probability Distributions

Each node in the network has an associated CPD defined using `TabularCPD`:

- **Marginal distributions**: F (Smoker), E (Exercise)
- **Conditional distributions**: P (Pressure|F,E), C (Cholesterol|E), A (Attack|P)

### 3. Model Validation

The model is validated to ensure:
- All CPDs sum to 1 for each parent configuration
- The graph is acyclic (DAG property)
- All dependencies are correctly specified

## Variable Elimination Algorithm

### Theoretical Foundation

Variable Elimination is an exact inference method for probabilistic graphical models. It works by:

1. **Representing queries as factor operations**
2. **Eliminating variables one at a time** through marginalization
3. **Using dynamic programming** to avoid redundant calculations

### Implementation

For a query like P(A|F=V), the algorithm:

1. Instantiate evidence: Set F=V in all relevant factors
2. Identify hidden variables: {E, P, C}
3. Choose elimination order: (C, E, P)
4. Perform factor operations:
   - Sum out C: $f_{\bar{C}}(E) = \sum_C f_C(C,E)$
   - Multiply factors: $f_{P\bar{C}}(P,E) = f_P(P,E) \times f_{\bar{C}}(E)$
   - Continue until only query variable remains

### Ordering Heuristics

Finding optimal elimination order is NP-HARD. We use these heuristics:

| Heuristic | Description |
|-----------|-------------|
| Min-neighbors | Eliminate variable with fewest neighbors |
| Min-weight | Minimize product of neighbor cardinalities |
| Min-fill | Minimize edges added to eliminate variable |

## Inference Queries

We implemented three key queries:

### Query 1: P(Attack | Smoker)
- **Question**: What's the heart attack probability for a smoker?
- **Method**: Eliminate {C, E, P} given F=V

### Query 2: P(High Pressure | Attack)
- **Question**: Given a heart attack occurred, what's the probability of high blood pressure?
- **Method**: Eliminate {F, E, C} given A=A, then normalize

### Query 3: P(Attack | Smoker, High Cholesterol)
- **Question**: What's the heart attack probability for a smoker with high cholesterol?
- **Method**: Eliminate {E, P} given F=V, C=A

## Validation Approach

Each manual calculation was verified against `pgmpy.inference.VariableElimination`:

```python
infer = VariableElimination(model)
result = infer.query(["A"], evidence={"F": "V"})
```

All results matched to 4 decimal places, confirming correct implementation.
