import numpy as np
import xgboost as xgb
# Import your custom objective function from your engine file
from src.xgboost_engine import financial_risk_objective 

def run_micro_batch_verification(X_train_engineered, y_train) -> bool:
    """Verifies if the network wrapper can perfectly memorize a tiny 5-row batch 
    using the custom financial risk objective.
    """
    print("\nRunning Karpathy Micro-Batch Verification Test...")

    # Isolate a tiny slice of exactly 5 records
    X_tiny = X_train_engineered.iloc[:5]
    y_tiny = y_train.iloc[:5]

    # Use your actual custom objective function here so it matches production math!
    overfit_tester = xgb.XGBRegressor(
        objective=financial_risk_objective,
        n_estimators=200,      # Increased estimators to ensure it can crush 5 rows
        max_depth=6, 
        learning_rate=0.3, 
        random_state=1337
    )

    overfit_tester.fit(X_tiny, y_tiny)
    tiny_preds = overfit_tester.predict(X_tiny)

    # Calculate your actual operational errors (predictions - actuals)
    errors = tiny_preds - y_tiny
    asymmetric_weights = np.where(errors > 0, 3.0, 1.0)
    tiny_asymmetric_mse = np.mean(asymmetric_weights * (errors**2))

    print(f"   - Tiny Batch Overfit Asymmetric Loss: {tiny_asymmetric_mse:.6f}")

    # Allow a slightly higher tolerance since asymmetric optimization settles conservatively
    if tiny_asymmetric_mse > 0.15:
        print(
            "   - Status: FAILED. Pipeline cannot memorize 5 rows. Check for bugs."
        )
        return False
    else:
        print(
            "   - Status: PASSED. Pipeline architecture is structurally sound."
        )
        return True

