import numpy as np
import pandas as pd
import pytest
from src.features import CassavaFeatureEngineer


def test_feature_engineer_prevents_zero_division():
    """Guarantees that your feature engine epsilon protects against a 0 water ratio."""
    # Create fake row in memory where the Water-to-Cassava Ratio is exactly 0
    mock_raw_data = pd.DataFrame(
        [
            {
                "Raw Material Storage Duration (Days)": 5,
                "Storage Temperature (°C)": 25,
                "Storage Humidity (%)": 60,
                "Cooking Temperature (°C)": 80,
                "Cooking Duration (Min)": 45,
                "Water-to-Cassava Ratio": 0.0,  # <--- The dangerous trap value
                "Alpha-Amylase Activity (U/g)": 120,
                "Alpha-Amylase Dosage (g/mL)": 1.5,
                "Liquefaction Temperature (°C)": 95,
                "Liquefaction Duration (Min)": 60,
            }
        ]
    )

    engineer = CassavaFeatureEngineer()

    # ACT & ASSERT: It must run without crashing with a ZeroDivisionError
    try:
        processed_df = engineer.transform(mock_raw_data)
        assert "Thermal_Hydration_Index" in processed_df.columns
        assert np.isfinite(processed_df["Thermal_Hydration_Index"].iloc[0])
    except ZeroDivisionError:
        pytest.fail("CassavaFeatureEngineer crashed on a zero water ratio!")

