import random
import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_ROWS = 1000  

# Cassava Properties
STARCH_MIN, STARCH_MAX = 20.0, 35.0          # %
MOISTURE_MIN, MOISTURE_MAX = 60.0, 65.0      # %

# Storage Conditions
STORAGE_DAYS_MIN, STORAGE_DAYS_MAX = 0, 7    # Days
STORAGE_TEMP_MIN, STORAGE_TEMP_MAX = 10.0, 20.0 # °C
STORAGE_HUMID_MIN, STORAGE_HUMID_MAX = 70.0, 90.0 # % 

# Cooking Stage
COOK_TEMP_MIN, COOK_TEMP_MAX = 85.0, 105.0   # °C
WATER_RATIO_MIN, WATER_RATIO_MAX = 2.0, 5.0  # Ratio
COOK_TIME_MIN, COOK_TIME_MAX = 30, 120       # Minutes

# Enzymatic Treatment (Liquefaction)
AMYLASE_ACT_MIN, AMYLASE_ACT_MAX = 50.0, 300.0 # U/g
AMYLASE_DOSE_MIN, AMYLASE_DOSE_MAX = 0.5, 5.0  # g or mL
LIQ_TEMP_MIN, LIQ_TEMP_MAX = 80.0, 95.0      # °C
LIQ_TIME_MIN, LIQ_TIME_MAX = 60, 180         # Minutes
PH_MIN, PH_MAX = 5.5, 6.5                    # pH units


def generate_columnar_data():
    """Generates independent data streams for every processing factor."""
    
    # Generating Float (Decimal) Columns
    starch_content = [random.uniform(STARCH_MIN, STARCH_MAX) for _ in range(NUM_ROWS)]
    moisture_content = [random.uniform(MOISTURE_MIN, MOISTURE_MAX) for _ in range(NUM_ROWS)]
    storage_temp = [random.uniform(STORAGE_TEMP_MIN, STORAGE_TEMP_MAX) for _ in range(NUM_ROWS)]
    storage_humid = [random.uniform(STORAGE_HUMID_MIN, STORAGE_HUMID_MAX) for _ in range(NUM_ROWS)]
    cook_temp = [random.uniform(COOK_TEMP_MIN, COOK_TEMP_MAX) for _ in range(NUM_ROWS)]
    water_ratio = [random.uniform(WATER_RATIO_MIN, WATER_RATIO_MAX) for _ in range(NUM_ROWS)]
    amylase_activity = [random.uniform(AMYLASE_ACT_MIN, AMYLASE_ACT_MAX) for _ in range(NUM_ROWS)]
    amylase_dosage = [random.uniform(AMYLASE_DOSE_MIN, AMYLASE_DOSE_MAX) for _ in range(NUM_ROWS)]
    liq_temp = [random.uniform(LIQ_TEMP_MIN, LIQ_TEMP_MAX) for _ in range(NUM_ROWS)]
    treatment_ph = [random.uniform(PH_MIN, PH_MAX) for _ in range(NUM_ROWS)]
    
    # Generating Integer (Whole Number) Columns
    storage_duration = [random.randint(STORAGE_DAYS_MIN, STORAGE_DAYS_MAX) for _ in range(NUM_ROWS)]
    cook_duration = [random.randint(COOK_TIME_MIN, COOK_TIME_MAX) for _ in range(NUM_ROWS)]
    liq_duration = [random.randint(LIQ_TIME_MIN, LIQ_TIME_MAX) for _ in range(NUM_ROWS)]

    return locals() # Temporarily grouping everything generated in this function

def calculate_yield(data_dict, num_rows):
    """
    Calculates the final Yield (%) column by allowing every single 
    independent chemical and process factor to influence the outcome.
    """
    yield_column = []
    
    for i in range(num_rows):
        # 1. Start with a realistic baseline yield (e.g., 75%)
        base_yield = 75.0
        
        # 2. Add POSITIVE influences (Higher values increase yield)
        starch_effect = (data_dict['starch_content'][i] - 20) * 0.4        # More starch = more raw material
        activity_effect = (data_dict['amylase_activity'][i] - 50) * 0.02   # Stronger enzymes speed up conversion
        dosage_effect = (data_dict['amylase_dosage'][i] - 0.5) * 0.8       # More enzyme breaks down more starch
        
        # 3. Add NEGATIVE influences / Penalties (Higher values decrease yield)
        # Starch degrades over storage days
        storage_effect = data_dict['storage_duration'][i] * -0.5           
        # Too much water dilutes the substrate matrix
        water_effect = (data_dict['water_ratio'][i] - 2.0) * -0.3          
        
        # 4. Add BELL-CURVE (Optimal Window) influences
        # Cooking is best around 95°C. Deviations lower the yield.
        cook_temp_deviation = abs(data_dict['cook_temp'][i] - 95.0)
        cook_temp_effect = cook_temp_deviation * -0.2
        
        # Liquefaction is best around 90°C.
        liq_temp_deviation = abs(data_dict['liq_temp'][i] - 90.0)
        liq_temp_effect = liq_temp_deviation * -0.2
        
        # Amylase prefers a pH of exactly 6.0. Moving away from 6.0 hurts yield.
        ph_deviation = abs(data_dict['treatment_ph'][i] - 6.0)
        ph_effect = ph_deviation * -3.0
        
        # Storage temp/humidity balance (higher combination stresses the roots)
        storage_stress = (data_dict['storage_temp'][i] * data_dict['storage_humid'][i]) / 1000
        storage_env_effect = storage_stress * -0.4

        # Duration stages (cooking & liquefaction need time, but over-processing burns energy)
        cook_time_effect = (data_dict['cook_duration'][i] - 30) * 0.02
        liq_time_effect = (data_dict['liq_duration'][i] - 60) * 0.01
        
        # Moisture balance
        moisture_effect = (data_dict['moisture_content'][i] - 60) * -0.1

        # 5. Combine all factors into the raw yield
        raw_yield = (
            base_yield + starch_effect + activity_effect + dosage_effect + 
            storage_effect + water_effect + cook_temp_effect + liq_temp_effect + 
            ph_effect + storage_env_effect + cook_time_effect + liq_time_effect + moisture_effect
        )
        
        # 6. Add "Process Noise" (Real-world experimental variance between -2% and +2%)
        noise = random.uniform(-2.0, 2.0)
        final_yield = raw_yield + noise
        
        # 7. Clamp the result between 0% and 100% so it remains physically possible
        final_yield = max(0.0, min(100.0, final_yield))
        
        yield_column.append(round(final_yield, 2))
        
    return yield_column



def save_dataset_to_csv(
    starch_content,
    moisture_content,
    storage_duration,
    storage_temp,
    storage_humid,
    cook_temp,
    water_ratio,
    cook_duration,
    amylase_activity,
    amylase_dosage,
    liq_temp,
    liq_duration,
    treatment_ph,
    yield_column,
    filename="cassava_yield_dataset.csv",
):
    """Zips independent process arrays together, rounds floats, and saves cleanly

    to a professional CSV file.
    """
    # 1. Define explicit, professional CSV headers
    headers = [
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
        "Predicted Yield (%)",
    ]

    print(f"Preparing to export data to {filename}...")

    try:
        # 2. Open file with correct encoding and newline configurations
        filepath = os.path.join(SCRIPT_DIR, filename)

        with open(filepath, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            # Write the top header row
            writer.writerow(headers)

            # 3. Zip columns together to iterate row-by-row
            all_columns = zip(
                starch_content,
                moisture_content,
                storage_duration,
                storage_temp,
                storage_humid,
                cook_temp,
                water_ratio,
                cook_duration,
                amylase_activity,
                amylase_dosage,
                liq_temp,
                liq_duration,
                treatment_ph,
                yield_column,
            )

            for row in all_columns:
                # 4. Professional formatting: Round floats to 2 decimal places on the fly
                formatted_row = [
                    round(val, 2) if isinstance(val, float) else val
                    for val in row
                ]
                writer.writerow(formatted_row)

        print(f"Success! Data pipeline complete. File saved as: '{filename}'")

    except IOError as e:
        print(f"System Error: Failed to write to disk. Details: {e}")

def main():
    data = generate_columnar_data()
    yield_column = calculate_yield(data, NUM_ROWS)

    save_dataset_to_csv(
        starch_content=data["starch_content"],
        moisture_content=data["moisture_content"],
        storage_duration=data["storage_duration"],
        storage_temp=data["storage_temp"],
        storage_humid=data["storage_humid"],
        cook_temp=data["cook_temp"],
        water_ratio=data["water_ratio"],
        cook_duration=data["cook_duration"],
        amylase_activity=data["amylase_activity"],
        amylase_dosage=data["amylase_dosage"],
        liq_temp=data["liq_temp"],
        liq_duration=data["liq_duration"],
        treatment_ph=data["treatment_ph"],
        yield_column=yield_column,
    )

main()