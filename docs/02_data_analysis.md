# Data Analysis

## Data Source

This project does not use a traditional dataset. Instead, the Bayesian Network is constructed using **expert-defined conditional probability distributions (CPDs)**. These probabilities represent medical knowledge about the relationships between risk factors and heart attacks.

## Conditional Probability Tables

### Prior Probabilities

**P(Smoker)**
| Smoker | Probability |
|--------|-------------|
| Yes (V) | 0.15 |
| No (F) | 0.85 |

**P(Exercise)**
| Exercise | Probability |
|----------|-------------|
| Yes (V) | 0.40 |
| No (F) | 0.60 |

### Conditional Probabilities

**P(High Pressure | Smoker, Exercise)**

| Smoker | Exercise | High Pressure | Probability |
|--------|----------|---------------|-------------|
| Yes | Yes | Yes | 0.45 |
| Yes | Yes | No | 0.55 |
| Yes | No | Yes | 0.95 |
| Yes | No | No | 0.05 |
| No | Yes | Yes | 0.05 |
| No | Yes | No | 0.95 |
| No | No | Yes | 0.55 |
| No | No | No | 0.45 |

**P(High Cholesterol | Exercise)**

| Exercise | High Cholesterol | Probability |
|----------|------------------|-------------|
| Yes | Yes | 0.40 |
| Yes | No | 0.60 |
| No | Yes | 0.80 |
| No | No | 0.20 |

**P(Heart Attack | High Pressure)**

| High Pressure | Heart Attack | Probability |
|---------------|--------------|-------------|
| Yes | Yes | 0.75 |
| Yes | No | 0.25 |
| No | Yes | 0.05 |
| No | No | 0.95 |

## Network Structure Analysis

The DAG represents the following causal assumptions:

1. **Smoking and lack of exercise independently contribute to high blood pressure**
2. **Lack of exercise leads to high cholesterol**
3. **High blood pressure directly increases heart attack risk**
4. **Cholesterol is a confounder** (affects both through exercise)

## Key Observations

1. **High-risk profile**: Smokers who don't exercise have 95% chance of high blood pressure
2. **Protective factors**: Non-smokers who exercise have only 5% chance of high blood pressure
3. **Direct causation**: High blood pressure gives 75% heart attack probability (vs 5% for normal)
4. **Exercise benefit**: Regular exercise reduces both high cholesterol (40% vs 80%) and indirectly reduces blood pressure
