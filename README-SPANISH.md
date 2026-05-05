# Predicción de Ataque Cardíaco

Modelo de Red Bayesiana aplicado a la predicción de ataques cardíacos usando métodos de eliminación de variables.

## Objetivos

- Predecir la probabilidad de sufrir un ataque cardíaco usando Redes Bayesianas.
- Aplicar el método de eliminación de variables en un grafo acíclico dirigido y comparar el resultado con los métodos implementados en pgmpy.

## Tecnologías

![Pgmpy](https://img.shields.io/badge/Pgmpy-2C8EBB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

## Hallazgos Clave

- La probabilidad de ataque cardíaco para un fumador es 57.5%, y aumenta a 62.75% cuando se combina con colesterol alto.
- La probabilidad de tener presión arterial alta dado un ataque cardíaco es 91.25%.
- Los cálculos manuales usando eliminación de variables coincidieron exactamente con los resultados de la librería pgmpy, validando la implementación.
- Encontrar el ordenamiento óptimo de las sumatorias sobre las variables ocultas es un problema NP-HARD, requiriendo heurísticas como min-neighbors.

## Curso

Introducción a la Ciencia de Datos

## Demo

Este proyecto tiene una demo disponible en el sitio web del portfolio.

## Repositorio

[https://github.com/Portfolio-KRV/heartattack](https://github.com/Portfolio-KRV/heartattack)
