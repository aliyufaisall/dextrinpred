import xgboost as xgb
import numpy as np
from src.config import RANDOM_SEED

# 1. Define the custom math function first
def financial_risk_objective(y_true, y_pred):
    """Custom asymmetric loss function to penalize over-predictions."""
    # XGBoost internal convention: (predictions - actuals)
    errors = y_pred - y_true 
    
    # Apply your exact 3.0 penalty weight when the model over-predicts (errors > 0)
    weights = np.where(errors > 0, 3.0, 1.0)
    
    # Compute first and second derivatives for gradient boosting
    gradient = 2 * weights * errors
    hessian = 2 * weights
    
    return gradient, hessian


# 2. Your production engine block can now safely reference it
def build_production_engine() -> xgb.XGBRegressor:
    """Initializes the primary XGBoost model architecture with robust structural regularizers."""
    return xgb.XGBRegressor(
        objective=financial_risk_objective,  # <--- This will now compile perfectly!
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,         # Shallow depth limits over-confident predictions
        subsample=0.8,       # Evaluates random blocks of rows to increase robustness
        colsample_bytree=0.8,# Evaluates random blocks of features per split
        random_state=RANDOM_SEED,
        n_jobs=-1,           # Utilizes all available processing cores
    )
