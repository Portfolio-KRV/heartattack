"""Main pipeline for training and evaluating the Heart Attack prediction model."""

import logging

from .evaluate import predict_heart_attack, run_validation_queries
from .logging_config import setup_logging
from .model import save_model, train_model

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the complete training and evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("Heart Attack Prediction - Bayesian Network")
    logger.info("=" * 60)

    # Build and save model
    logger.info("1. Building model...")
    model = train_model()
    save_model(model)

    # Run validation queries
    logger.info("2. Running validation queries...")
    validation_results = run_validation_queries(model)

    all_passed = True
    for result in validation_results:
        status = "PASS" if result["passed"] else "FAIL"
        logger.info("   %s: Expected=%.4f, Actual=%.4f [%s]",
                   result['query'], result['expected'], result['actual'], status)
        if not result["passed"]:
            all_passed = False

    if all_passed:
        logger.info("   All validation queries passed!")
    else:
        logger.warning("   WARNING: Some validation queries failed!")

    # Run example predictions
    logger.info("3. Example predictions...")

    examples = [
        {"smoker": True, "exercise": None, "high_cholesterol": None, "high_pressure": None},
        {"smoker": True, "exercise": False, "high_cholesterol": True, "high_pressure": None},
        {"smoker": False, "exercise": True, "high_cholesterol": False, "high_pressure": None},
    ]

    for i, example in enumerate(examples, 1):
        result = predict_heart_attack(model, **example)
        logger.info("   Example %d: %s", i, example)
        logger.info("   -> %s (Risk: %s)", result['prediction'], result['risk_level'])
        logger.info("   -> P(Attack) = %.2f%%", result['probability_attack'] * 100)

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_logging()
    main()
