"""FastAPI application for Heart Attack prediction."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src import __version__
from src.api_common import create_app, limiter, register_error_handlers
from src.evaluate import predict_heart_attack, query
from src.exceptions import (
    ConfigurationError,
    DataError,
    DataValidationError,
    ModelError,
    ModelLoadError,
    ModelNotLoadedError,
)
from src.model import get_or_create_model

from .schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    QueryRequest,
    QueryResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    model = get_or_create_model()
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down")


app = create_app(
    title="Heart Attack Prediction API",
    description="""
    API for predicting heart attack probability using Bayesian Networks.

    This model uses the Variable Elimination algorithm for exact probabilistic inference
    on a directed acyclic graph (DAG) representing causal relationships between
    risk factors and heart attack occurrence.

    ## Variables

    - **Smoker (F)**: Whether the patient smokes
    - **Exercise (E)**: Whether the patient exercises regularly
    - **High Pressure (P)**: Whether the patient has high blood pressure
    - **High Cholesterol (C)**: Whether the patient has high cholesterol
    - **Heart Attack (A)**: Target variable for prediction

    ## Usage

    Provide any combination of known risk factors to get a heart attack probability estimate.
    """,
    version=__version__,
    lifespan=lifespan,
)
register_error_handlers(
    app,
    {
        ModelNotLoadedError: (503, "model_not_loaded"),
        ModelLoadError: (500, "model_load_error"),
        ModelError: (500, "model_error"),
        DataValidationError: (422, "validation_error"),
        DataError: (500, "data_error"),
        ConfigurationError: (500, "configuration_error"),
    },
    expose_message=(DataValidationError,),
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version=__version__,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("20/minute")
async def predict(request: Request, body: PredictionRequest):
    """Predict heart attack probability given risk factors.

    Provide any combination of known risk factors. Unspecified factors
    will be marginalized out using the model's prior distributions.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = predict_heart_attack(
        model,
        smoker=body.smoker,
        exercise=body.exercise,
        high_cholesterol=body.high_cholesterol,
        high_pressure=body.high_pressure,
    )

    return PredictionResponse(**result)


@app.post("/query", response_model=QueryResponse, tags=["Advanced"])
@limiter.limit("20/minute")
async def custom_query(request: Request, body: QueryRequest):
    """Execute a custom probabilistic query on the Bayesian Network.

    This endpoint allows direct queries using variable codes:
    - A: Heart Attack
    - P: High Pressure
    - C: High Cholesterol
    - F: Smoker
    - E: Exercise

    States:
    - For F, E: "V" (Yes), "F" (No)
    - For A, P, C: "A" (Yes), "B" (No)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    valid_variables = {"A", "P", "C", "F", "E"}
    if body.target not in valid_variables:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target variable. Must be one of: {valid_variables}",
        )

    try:
        if body.evidence:
            for var in body.evidence:
                if var not in valid_variables:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid evidence variable '{var}'. Must be one of: {valid_variables}",
                    )

        result = query(model, body.target, body.evidence)
        return QueryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing query: {e!s}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail="Invalid query parameters. Please check your input.",
        ) from e


@app.get("/variables", tags=["Info"])
@limiter.limit("60/minute")
async def get_variables(request: Request):
    """Get information about model variables."""
    from src.config import VARIABLES
    return VARIABLES
