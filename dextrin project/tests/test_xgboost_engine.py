import numpy as np
from src.xgboost_engine import financial_risk_objective


def test_asymmetric_loss_penalizes_overprediction():
    """Guarantees that the gradient math strictly punishes overpromising."""
    y_true = np.array([50.0])

    # Case A: Model over-predicted by 5 units (y_pred = 55) -> High business risk!
    grad_over, _ = financial_risk_objective(y_true, np.array([55.0]))

    # Case B: Model under-predicted by 5 units (y_pred = 45) -> Safe conservative guess
    grad_under, _ = financial_risk_objective(y_true, np.array([45.0]))

    # The gradient magnitude for over-predicting must be exactly 3 times larger
    assert abs(grad_over[0]) == 3.0 * abs(grad_under[0])
