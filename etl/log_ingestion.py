# etl/log_ingestion.py
import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def setup_database(db_path, schema_path):
    """
    Initializes the SQLite database using the schema SQL file.
    """
    print(f"Initializing database at {db_path} using schema from {schema_path}...")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
        
    cursor.executescript(schema_sql)
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def simulate_logs(days=60, logs_per_day=500, seed=42):
    """
    Simulates model inference logs over a set number of days.
    Gradually introduces feature drift and label drift in the second half of the period.
    """
    np.random.seed(seed)
    start_date = datetime.now() - timedelta(days=days)
    
    records = []
    
    gender_options = ['Female', 'Male']
    gender_probs = [0.48, 0.52]
    
    age_options = ['18-25', '26-35', '36-50', '50+']
    age_probs = [0.20, 0.35, 0.30, 0.15]
    
    print(f"Simulating {logs_per_day} logs per day for {days} days...")
    
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        
        # Calculate drift factors (drift starts on Day 30 and scales up to 1.0 by Day 60)
        if day < 30:
            drift_factor = 0.0
        else:
            drift_factor = (day - 30) / 30.0
            
        for _ in range(logs_per_day):
            gender = np.random.choice(gender_options, p=gender_probs)
            age_group = np.random.choice(age_options, p=age_probs)
            
            # Feature 1 (e.g. standardized income score)
            # Baseline: Female mean = 4.8, Male mean = 5.0, Std = 1.2
            # Under drift: Female mean drops more, Male mean drops less. Widens inequality.
            f1_mean_female = 4.8 - (1.1 * drift_factor)
            f1_mean_male = 5.0 - (0.4 * drift_factor)
            f1_mean = f1_mean_female if gender == 'Female' else f1_mean_male
            input_feature_1 = np.random.normal(f1_mean, 1.2)
            
            # Feature 2 (e.g. credit utilization factor)
            # Baseline: Mean = 650, Std = 80
            # Under drift: Mean drops by 70 points for everyone
            f2_mean = 650.0 - (70.0 * drift_factor)
            input_feature_2 = np.random.normal(f2_mean, 80.0)
            
            # Model Score Logic (Logistic Regression Simulation)
            # Baseline model weight vector
            z = 0.45 * input_feature_1 + 0.006 * (input_feature_2 - 500) - 1.0
            raw_score = 1.0 / (1.0 + np.exp(-z))
            
            # Decision threshold
            predicted_class = 1 if raw_score >= 0.50 else 0
            
            # Ground truth generation (repay or default)
            # In early days: the model is highly calibrated.
            # In late days: economic environment decays, defaults rise (Concept Drift),
            # causing model accuracy to degrade as the true boundary shifts.
            if day < 30:
                true_z = z + np.random.normal(0, 0.15)
                actual_prob = 1.0 / (1.0 + np.exp(-true_z))
                actual_label = 1 if actual_prob >= 0.50 else 0
            else:
                # The relationship shifts: higher chance of failure (0) for the same scores
                true_z = z - (0.95 * drift_factor) + np.random.normal(0, 0.40)
                actual_prob = 1.0 / (1.0 + np.exp(-true_z))
                # Threshold also changes, creating accuracy drop
                actual_label = 1 if actual_prob >= 0.55 else 0
                
            # Timestamp generation (spread throughout the day)
            hour = np.random.randint(0, 24)
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            prediction_timestamp = current_date.replace(hour=hour, minute=minute, second=second)
            
            records.append((
                prediction_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                gender,
                age_group,
                float(raw_score),
                int(predicted_class),
                int(actual_label),
                float(input_feature_1),
                float(input_feature_2)
            ))
            
    df = pd.DataFrame(records, columns=[
        'prediction_timestamp', 'gender', 'age_group', 'raw_score',
        'predicted_class', 'actual_label', 'input_feature_1', 'input_feature_2'
    ])
    
    return df

def save_to_db(df, db_path):
    """
    Saves the simulated records to the inference_logs table in SQLite.
    """
    print(f"Saving {len(df)} records to inference_logs...")
    conn = sqlite3.connect(db_path)
    
    # We append to avoid dropping the structure
    df.to_sql('inference_logs', conn, if_exists='append', index=False)
    conn.close()
    print("Inference logs saved successfully.")

if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_file = os.path.join(project_dir, 'data', 'observability.db')
    schema_file = os.path.join(project_dir, 'db', 'schema.sql')
    
    # 1. Initialize DB schema
    setup_database(db_file, schema_file)
    
    # 2. Simulate 60 days of data
    df_logs = simulate_logs(days=60, logs_per_day=500)
    
    # 3. Save to database
    save_to_db(df_logs, db_file)
    print("ETL Log Ingestion process completed.")
