import os

# Root & Data Paths
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  # Points to project root
RAW_DATA_PATH = os.path.join(
    BASE_DIR, "data", "cassava_yield_dataset.csv"
)

# Phase 1: Target Definition and Global Reproducibility Hyperparameters
TARGET_COLUMN = "Predicted Yield (%)"
RANDOM_SEED = 1337  #

# Column Verification List
EXPECTED_FEATURES = [
    "Cassava Starch Content (%)",
    "Cassava Moisture Content (%)",
    "Raw Material Storage Duration (Days)",
    "Storage Temperature (°C)",
    "Storage Humidity (%)",
    "Cooking Temperature (°C)",
    "Water-to-Cassava Ratio",
    "Cooking Duration (Min)",
    "Alpha-Amylase Activity (U/g)",
    "Alpha-Amylase Dosage (g/mL)",
    "Liquefaction Temperature (°C)",
    "Liquefaction Duration (Min)",
    "pH During Enzymatic Treatment",
]
