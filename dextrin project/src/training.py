import joblib
import os
from sklearn.pipeline import Pipeline  # <--- 1. IMPORT THE PIPELINE WRAPPER
from src.baseline_model import train_dumb_baseline
from src.config import RAW_DATA_PATH
from src.evaluation import compute_production_metrics
from src.features import CassavaFeatureEngineer
from src.loader import load_and_verify_data, split_to_three_way
from src.sanity_checks import run_micro_batch_verification
from src.xgboost_engine import build_production_engine


if __name__ == "__main__":
    print("Initializing Startup Engine Pipeline...")

    # Phase 1 & 2: Ingestion & Three-Way Splitting
    raw_df = load_and_verify_data(RAW_DATA_PATH)
    X_train, X_val, X_lockbox, y_train, y_val, y_lockbox = split_to_three_way(raw_df)

    print("\nPhase 1 & Phase 2 Complete. Data splits isolated.")

    # =========================================================================
    # THE CRACKED ENGINEER FIX: EMBED EVERYTHING INTO A UNIFIED PIPELINE
    # =========================================================================
    print("\nAssembling leakage-safe production pipeline wrapper...")
    production_pipeline = Pipeline([
        ('feature_engineering', CassavaFeatureEngineer()), # Steps 1-5 math
        ('production_engine', build_production_engine())   # Custom Asymmetric XGBoost
    ])
    
    # We still engineer features separately for the baseline and micro-batch sanity test
    engineer = CassavaFeatureEngineer()
    X_train_eng = engineer.fit_transform(X_train)
    X_val_eng = engineer.transform(X_val)

    # Phase 3: Train Simple Baseline Model
    print("\nTraining Phase 3 Simple Baseline Model...")
    baseline_tree = train_dumb_baseline(X_train_eng, y_train)
    baseline_val_preds = baseline_tree.predict(X_val_eng)
    baseline_metrics = compute_production_metrics(y_val, baseline_val_preds)
    
    # Phase 4: Run Karpathy Micro-Batch Overfit Check
    pipeline_is_safe = run_micro_batch_verification(X_train_eng, y_train)

    if pipeline_is_safe:
        # Train the entire pipeline on the RAW training data. 
        # The Pipeline object automatically transforms the data safely under the hood.
        print("\nTraining Production Pipeline Wrapper on raw dataset...")
        production_pipeline.fit(X_train, y_train) # <--- 2. FIT THE ENTIRE PIPELINE AT ONCE
        # SAVE THE TRAINED PIPELINE TO A SPECIFIC FOLDER
        # Define your target folder and file name
        output_dir = "models"
        model_filename = os.path.join(output_dir, "production_pipeline.joblib")
        
        # Safely create the folder if it does not exist yet
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving fully trained pipeline to '{model_filename}'...")
        joblib.dump(production_pipeline, model_filename)
        print("Model saved successfully!")

        # Predict using RAW validation data. No manually tracking separate transformations.
        val_predictions = production_pipeline.predict(X_val) # <--- 3. PREDICT ON RAW DATA SAFELY
        production_metrics = compute_production_metrics(y_val, val_predictions)

        print("\n==================================================")
        print("FINAL PIPELINE AUDIT REPORT")
        print("==================================================")
        print("Baseline Decision Tree:")
        print(f"   - Standard RMSE:          {baseline_metrics['RMSE']:.4f}")
        print(f"   - Financial Risk Loss:    {baseline_metrics['Financial_Risk_Loss']:.4f}")
        print(f"   - Over-Prediction Rate:   {baseline_metrics['Over_Prediction_Rate']:.1f}%")
        print("--------------------------------------------------")
        print("Production Pipeline Wrapper:")
        print(f"   - Standard RMSE:          {production_metrics['RMSE']:.4f}")
        print(f"   - Financial Risk Loss:    {production_metrics['Financial_Risk_Loss']:.4f}")
        print(f"   - Over-Prediction Rate:   {production_metrics['Over_Prediction_Rate']:.1f}%")
        print("==================================================")
    else:
        print("\n[CRITICAL FAILURE] Pipeline halted. Micro-batch verification failed.")


