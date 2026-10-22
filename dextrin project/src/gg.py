import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# =====================================================================
# IMPORTANT: Keep your custom class defined right here at the top 
# so joblib can see it when it unpickles 'production_pipeline.joblib'
# =====================================================================
class CassavaFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_out["Storage_Spoilage_Penalty"] = (
            X_out["Raw Material Storage Duration (Days)"]
            * X_out["Storage Temperature (°C)"]
            * X_out["Storage Humidity (%)"]
        )
        cooking_heat_load = (
            X_out["Cooking Temperature (°C)"]
            * X_out["Cooking Duration (Min)"]
        )
        X_out["Thermal_Hydration_Index"] = cooking_heat_load / (
            X_out["Water-to-Cassava Ratio"] + 1e-5
        )
        X_out["Total_Enzyme_Power_Pool"] = (
            X_out["Alpha-Amylase Activity (U/g)"]
            * X_out["Alpha-Amylase Dosage (g/mL)"]
        )
        X_out["Liquefaction_Heat_Stress"] = (
            X_out["Liquefaction Temperature (°C)"]
            * X_out["Liquefaction Duration (Min)"]
        )
        return X_out

# 1. Environment and Path setup
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

@st.cache_resource
def load_model():
    model_path = root_dir / "models" / "production_pipeline.joblib"
    return joblib.load(model_path)

model = load_model()

# 2. Main Title Layout
st.title("Dextrin Yield Predictor")

# 3. Create user input widgets (Sliders)
st.subheader("Input Processing Parameters")

with st.container():
    st.markdown("### Raw Material Properties")
    starch_content = st.slider("Cassava Starch Content (%)", 10.0, 45.0, 30.0, 0.1)
    moisture_content = st.slider("Cassava Moisture Content (%)", 40.0, 80.0, 65.0, 0.1)

    st.markdown("### Storage Conditions")
    storage_duration = st.slider("Raw Material Storage Duration (Days)", 0, 30, 3, 1)
    storage_temp = st.slider("Storage Temperature (°C)", 15.0, 45.0, 28.0, 0.5)
    storage_humidity = st.slider("Storage Humidity (%)", 30.0, 95.0, 75.0, 1.0)

    st.markdown("### Gelatinization & Cooking")
    cooking_temp = st.slider("Cooking Temperature (°C)", 50.0, 120.0, 85.0, 0.5)
    water_ratio = st.slider("Water-to-Cassava Ratio", 1.0, 10.0, 3.5, 0.1)
    cooking_duration = st.slider("Cooking Duration (Min)", 5, 180, 45, 1)

    st.markdown("### Enzymatic Treatment & Liquefaction")
    enzyme_activity = st.slider("Alpha-Amylase Activity (U/g)", 100.0, 5000.0, 1500.0, 10.0)
    enzyme_dosage = st.slider("Alpha-Amylase Dosage (g/mL)", 0.001, 0.100, 0.015, 0.001, format="%.3f")
    liquefaction_temp = st.slider("Liquefaction Temperature (°C)", 50.0, 110.0, 90.0, 0.5)
    liquefaction_duration = st.slider("Liquefaction Duration (Min)", 10, 240, 90, 1)
    process_ph = st.slider("pH During Enzymatic Treatment", 3.0, 9.0, 6.0, 0.1)

# 4. Construct DataFrame matching your exact long-text column criteria
features_dict = {
    "Cassava Starch Content (%)": [starch_content],
    "Cassava Moisture Content (%)": [moisture_content],
    "Raw Material Storage Duration (Days)": [storage_duration],
    "Storage Temperature (°C)": [storage_temp],
    "Storage Humidity (%)": [storage_humidity],
    "Cooking Temperature (°C)": [cooking_temp],
    "Water-to-Cassava Ratio": [water_ratio],
    "Cooking Duration (Min)": [cooking_duration],
    "Alpha-Amylase Activity (U/g)": [enzyme_activity],
    "Alpha-Amylase Dosage (g/mL)": [enzyme_dosage],
    "Liquefaction Temperature (°C)": [liquefaction_temp],
    "Liquefaction Duration (Min)": [liquefaction_duration],
    "pH During Enzymatic Treatment": [process_ph],
}

input_df = pd.DataFrame(features_dict)

# 5. Prediction Execution
if st.button("Calculate Yield Prediction", type="primary"):
    # Since your pipeline contains 'CassavaFeatureEngineer', it will run 
    # .transform(input_df) internally. We just pass input_df directly.
    prediction = model.predict(input_df)
    
    # 6. Display Output
    st.success("Calculation Complete!")
    
    # Extract prediction float safely
    if hasattr(prediction, "__len__") and not isinstance(prediction, (str, bytes)):
        final_val = prediction[0]
    else:
        final_val = prediction
        
    st.metric(label="Estimated Cassava Yield", value=f"{final_val:.2f}")
