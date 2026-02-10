# analytics/audit_engine.py
import os
import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def run_fairness_and_drift_audit(db_path):
    """
    Performs observability and fairness auditing on inference logs.
    Saves daily audit summaries to the database.
    """
    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # Load all inference logs
    df = pd.read_sql_query("SELECT * FROM inference_logs", conn)
    if len(df) == 0:
        print("No logs found in inference_logs. Please run log_ingestion.py first.")
        conn.close()
        return
    
    # Preprocess date and time
    df['prediction_timestamp'] = pd.to_datetime(df['prediction_timestamp'])
    df['audit_date'] = df['prediction_timestamp'].dt.strftime('%Y-%m-%d')
    
    # Get sorted unique dates
    unique_dates = sorted(df['audit_date'].unique())
    print(f"Found {len(unique_dates)} unique days of data.")
    
    # Define baseline period: first 5 days
    baseline_days = unique_dates[:5]
    baseline_df = df[df['audit_date'].isin(baseline_days)]
    
    # Compute baseline reference metrics
    baseline_f1 = baseline_df['input_feature_1'].values
    baseline_accuracy = (baseline_df['predicted_class'] == baseline_df['actual_label']).mean()
    
    print(f"Baseline established (Days {baseline_days[0]} to {baseline_days[-1]}):")
    print(f"  - Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"  - Baseline Feature 1 Size: {len(baseline_f1)}")
    
    audit_results = []
    
    for date in unique_dates:
        day_df = df[df['audit_date'] == date]
        
        # 1. Feature Drift: Kolmogorov-Smirnov Test on input_feature_1
        day_f1 = day_df['input_feature_1'].values
        ks_stat, p_value = ks_2samp(baseline_f1, day_f1)
        
        # 2. Fairness Metric: Disparate Impact (DI) Ratio (Female vs Male)
        # Privileged Group = Male, Unprivileged Group = Female
        male_df = day_df[day_df['gender'] == 'Male']
        female_df = day_df[day_df['gender'] == 'Female']
        
        sr_male = (male_df['predicted_class'] == 1).mean() if len(male_df) > 0 else 0.0
        sr_female = (female_df['predicted_class'] == 1).mean() if len(female_df) > 0 else 0.0
        
        # Handle division by zero
        if sr_male > 0:
            disparate_impact_ratio = sr_female / sr_male
        else:
            disparate_impact_ratio = 1.0 if sr_female == 0 else 99.0
            
        # 3. Fairness Metric: Demographic Parity Ratio (DPR) across Age Groups
        # DPR = min(Selection Rate) / max(Selection Rate) across all age subgroups
        age_groups = day_df['age_group'].unique()
        age_selection_rates = {}
        for age in age_groups:
            age_sub_df = day_df[day_df['age_group'] == age]
            sr_age = (age_sub_df['predicted_class'] == 1).mean() if len(age_sub_df) > 0 else 0.0
            age_selection_rates[age] = sr_age
            
        if len(age_selection_rates) > 0:
            min_sr = min(age_selection_rates.values())
            max_sr = max(age_selection_rates.values())
            demographic_parity_ratio = min_sr / max_sr if max_sr > 0 else 1.0
        else:
            demographic_parity_ratio = 1.0
            
        # 4. Performance Decay: Accuracy Degradation
        day_accuracy = (day_df['predicted_class'] == day_df['actual_label']).mean()
        accuracy_degradation = baseline_accuracy - day_accuracy
        
        audit_results.append({
            'audit_date': date,
            'disparate_impact_ratio': float(disparate_impact_ratio),
            'demographic_parity_ratio': float(demographic_parity_ratio),
            'drift_p_value': float(p_value),
            'accuracy_degradation': float(accuracy_degradation)
        })
        
    audit_df = pd.DataFrame(audit_results)
    
    # Save results to sqlite database
    print("Writing audited metrics to fairness_audit_history...")
    
    # Clean old audit run to prevent duplicate constraint violation
    cursor = conn.cursor()
    cursor.execute("DELETE FROM fairness_audit_history")
    conn.commit()
    
    # Save
    audit_df.to_sql('fairness_audit_history', conn, if_exists='append', index=False)
    conn.close()
    
    print(f"Fairness & Drift Audit completed. Logged {len(audit_df)} records.")

if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_file = os.path.join(project_dir, 'data', 'observability.db')
    
    run_fairness_and_drift_audit(db_file)
