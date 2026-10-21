import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CassavaFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering pipeline built on your business logic.

    Calculates compounding degradation penalties from storage conditions.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        # Your custom formula: Yield reduces as duration, temperature, or humidity increases
        X_out["Storage_Spoilage_Penalty"] = (
            X_out["Raw Material Storage Duration (Days)"]
            * X_out["Storage Temperature (°C)"]
            * X_out["Storage Humidity (%)"]
        )

        # 2. Base Cooking Heat Load Proxy (Curve mapping)
        cooking_heat_load = (
            X_out["Cooking Temperature (°C)"]
            * X_out["Cooking Duration (Min)"]
        )

        # 3. Your Cooking/Water Logic: Heat load balanced against the hydration ratio
        # Small epsilon (1e-5) added to denominator to guarantee no division-by-zero crashes
        X_out["Thermal_Hydration_Index"] = cooking_heat_load / (
            X_out["Water-to-Cassava Ratio"] + 1e-5
        )
        # 4. ENZYME LOGIC: Total kinetic work capacity from worker strength (Activity) × worker count (Dosage)
        X_out["Total_Enzyme_Power_Pool"] = (
            X_out["Alpha-Amylase Activity (U/g)"]
            * X_out["Alpha-Amylase Dosage (g/mL)"]
        )
        # 5. LIQUEFACTION LOGIC: Total thermal stress applied to enzymes over time
        X_out["Liquefaction_Heat_Stress"] = (
            X_out["Liquefaction Temperature (°C)"]
            * X_out["Liquefaction Duration (Min)"]
        )
        return X_out