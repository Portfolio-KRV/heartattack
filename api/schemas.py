"""Pydantic schemas for API request/response validation."""


from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for heart attack prediction.

    All fields are optional. The model will use available evidence
    to compute conditional probabilities.
    """

    smoker: bool | None = Field(
        None,
        description="Whether the patient is a smoker",
        examples=[True],
    )
    exercise: bool | None = Field(
        None,
        description="Whether the patient exercises regularly",
        examples=[True],
    )
    high_cholesterol: bool | None = Field(
        None,
        description="Whether the patient has high cholesterol",
        examples=[False],
    )
    high_pressure: bool | None = Field(
        None,
        description="Whether the patient has high blood pressure",
        examples=[False],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "smoker": True,
                    "exercise": False,
                    "high_cholesterol": True,
                    "high_pressure": None,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for heart attack prediction."""

    prediction: str = Field(
        ...,
        description="Prediction result: 'Heart Attack Likely' or 'Heart Attack Unlikely'",
        examples=["Heart Attack Likely"],
    )
    probability_attack: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of heart attack (0-1)",
        examples=[0.575],
    )
    probability_no_attack: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of no heart attack (0-1)",
        examples=[0.425],
    )
    risk_level: str = Field(
        ...,
        description="Risk level: 'Low', 'Medium', or 'High'",
        examples=["Medium"],
    )
    evidence_provided: dict = Field(
        ...,
        description="The evidence used for prediction",
    )


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    version: str = Field(..., examples=["1.0.0"])


class QueryRequest(BaseModel):
    """Request schema for custom probabilistic queries."""

    target: str = Field(
        ...,
        description="Target variable to query (A, P, C, F, E)",
        examples=["A"],
    )
    evidence: dict | None = Field(
        None,
        description="Evidence variables as {variable: state}",
        examples=[{"F": "V", "C": "A"}],
    )


class QueryResponse(BaseModel):
    """Response schema for custom queries."""

    variable: str = Field(..., description="Queried variable")
    variable_name: str = Field(..., description="Human-readable variable name")
    evidence: dict = Field(..., description="Evidence used")
    probabilities: dict = Field(..., description="State probabilities")
