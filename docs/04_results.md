# Results

## Query Results

### Query 1: Heart Attack Probability Given Smoker

**P(A | F = V)**

| Heart Attack | Probability |
|--------------|-------------|
| Yes (A) | **57.50%** |
| No (B) | 42.50% |

**Interpretation**: A smoker has a 57.5% probability of suffering a heart attack before age 70. This is significantly higher than the baseline population risk.

---

### Query 2: High Blood Pressure Given Heart Attack

**P(P | A = A)**

| High Pressure | Probability |
|---------------|-------------|
| Yes (A) | **91.25%** |
| No (B) | 8.75% |

**Interpretation**: If a patient has had a heart attack, there's a 91.25% probability they have high blood pressure. This demonstrates the strong diagnostic relationship between blood pressure and heart attacks.

---

### Query 3: Heart Attack Probability Given Smoker and High Cholesterol

**P(A | F = V, C = A)**

| Heart Attack | Probability |
|--------------|-------------|
| Yes (A) | **62.75%** |
| No (B) | 37.25% |

**Interpretation**: A smoker with high cholesterol has a 62.75% probability of heart attack, which is higher than being a smoker alone (57.5%). The combination of risk factors increases overall risk.

---

## Validation Results

All manual calculations using the Variable Elimination method were verified against the `pgmpy` library implementation:

| Query | Manual Result | pgmpy Result | Match |
|-------|---------------|--------------|-------|
| P(A\|F=V) | 0.5750 | 0.5750 | Yes |
| P(P\|A=A) | 0.9125 | 0.9125 | Yes |
| P(A\|F=V,C=A) | 0.6275 | 0.6275 | Yes |

## Key Findings

1. **Smoking significantly increases heart attack risk** (57.5% vs baseline)

2. **High blood pressure is a strong indicator** of heart attack (91.25% posterior probability)

3. **Risk factors compound**: Smoker + high cholesterol = 62.75% risk, higher than smoking alone

4. **Variable elimination produces exact results** matching library implementations

5. **The ordering of variable elimination affects computational efficiency** but not the final result

## Practical Applications

This model can be used to:

1. **Risk Assessment**: Calculate personalized heart attack probability given known risk factors
2. **Intervention Planning**: Identify which factors to target for maximum risk reduction
3. **Diagnostic Support**: Given symptoms, infer likely underlying conditions
4. **Patient Education**: Demonstrate how lifestyle changes affect health outcomes
