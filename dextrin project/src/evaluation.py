import numpy as np


def compute_production_metrics(y_true, y_pred):
    """Calculates standard ML metrics alongside a custom business-risk asymmetric metric."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    errors = y_pred - y_true

    # Standard Symmetric Metric
    rmse = np.sqrt(np.mean(errors**2))

    # Asymmetric Financial Loss: Penalize over-prediction heavily
    # If error > 0 (over-predicted), multiply penalty weight by 3.0
    asymmetric_weights = np.where(errors > 0, 3.0, 1.0)
    financial_risk_score = np.mean(asymmetric_weights * (errors**2))

    # Calculate exactly how many times the engine over-predicted
    over_prediction_count = np.sum(errors > 0)
    total_predictions = len(errors)
    over_prediction_rate = (over_prediction_count / total_predictions) * 100

    return {
        "RMSE": rmse,
        "Financial_Risk_Loss": financial_risk_score,
        "Over_Prediction_Rate": over_prediction_rate,
    }
