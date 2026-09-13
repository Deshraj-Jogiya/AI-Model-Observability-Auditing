# viz/dashboard_export.py
import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from datetime import datetime

def generate_governance_dashboard(db_path, output_image_path):
    """
    Reads SQLite data and exports a high-quality visualization panel.
    """
    print(f"Loading data for dashboard from {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # 1. Load data
    df_logs = pd.read_sql_query("SELECT * FROM inference_logs", conn)
    df_audit = pd.read_sql_query("SELECT * FROM fairness_audit_history", conn)
    conn.close()
    
    if len(df_logs) == 0 or len(df_audit) == 0:
        print("Data is missing. Run log_ingestion.py and audit_engine.py first.")
        return

    # Parse dates
    df_logs['prediction_timestamp'] = pd.to_datetime(df_logs['prediction_timestamp'])
    df_logs['audit_date'] = df_logs['prediction_timestamp'].dt.strftime('%Y-%m-%d')
    df_audit['audit_date'] = pd.to_datetime(df_audit['audit_date'])
    
    # Style configuration for a premium dark-slate aesthetic
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.color'] = '#F8FAFC'
    plt.rcParams['axes.labelcolor'] = '#94A3B8'
    plt.rcParams['xtick.color'] = '#94A3B8'
    plt.rcParams['ytick.color'] = '#94A3B8'
    
    # Create the figure
    fig = plt.figure(figsize=(20, 12), facecolor='#0F172A')
    fig.suptitle('Tableau AI Governance & Observability Dashboard\nModel Performance, Feature Drift, and Fairness Audit Panel', 
                 fontsize=22, color='#F8FAFC', weight='bold', y=0.96)
    
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # --- PANEL A: Input Data Drift Trend Line ---
    ax_drift = fig.add_subplot(gs[0, 0])
    ax_drift.set_facecolor('#1E293B')
    
    # Plot daily p-value
    sns.lineplot(data=df_audit, x='audit_date', y='drift_p_value', color='#38BDF8', linewidth=2.5, ax=ax_drift, label='KS test p-value')
    # Threshold line for significance (p = 0.05)
    ax_drift.axhline(y=0.05, color='#EF4444', linestyle='--', linewidth=1.5, label='Significance Threshold (0.05)')
    
    ax_drift.set_yscale('log')
    ax_drift.set_title('A. Input Data Drift Trend (KS Test p-value & PSI over time)', fontsize=14, pad=12, weight='bold', color='#F8FAFC')
    ax_drift.set_xlabel('Audit Date', fontsize=11)
    ax_drift.set_ylabel('p-value (Log Scale)', fontsize=11)
    ax_drift.grid(True, which="both", ls=":", color='#334155', alpha=0.5)

    # PSI on its own linear axis -- a complementary severity score (0.1/0.25
    # conventional thresholds) rather than the p-value's binary significant call.
    ax_psi = ax_drift.twinx()
    sns.lineplot(data=df_audit, x='audit_date', y='psi_score', color='#FBBF24', linewidth=2, ax=ax_psi, label='PSI')
    ax_psi.axhline(y=0.25, color='#FBBF24', linestyle=':', linewidth=1, alpha=0.7)
    ax_psi.set_ylabel('PSI', fontsize=11, color='#FBBF24')
    ax_psi.tick_params(axis='y', colors='#FBBF24')
    ax_drift.legend(facecolor='#1E293B', edgecolor='#475569', labelcolor='#F8FAFC')
    
    # Format x-axis dates
    fig.autofmt_xdate(rotation=30)
    
    # --- PANEL B: Disparate Impact Ratio by Subgroup ---
    # We will compute the disparate impact of age groups and gender over the last 15 drifted days.
    # Disparate impact is calculated relative to a reference group.
    # For gender: Reference is Male. For Age Group: Reference is '36-50' (highest selection rate historically).
    ax_fairness = fig.add_subplot(gs[0, 1])
    ax_fairness.set_facecolor('#1E293B')
    
    recent_df = df_logs[df_logs['audit_date'] >= df_audit['audit_date'].dt.strftime('%Y-%m-%d').iloc[-15]]
    
    # Calculate selection rates by group
    subgroups = []
    selection_rates = []
    
    # Gender rates
    for g in ['Female', 'Male']:
        sub_df = recent_df[recent_df['gender'] == g]
        sr = (sub_df['predicted_class'] == 1).mean()
        subgroups.append(g)
        selection_rates.append(sr)
        
    # Age rates
    for a in ['18-25', '26-35', '36-50', '50+']:
        sub_df = recent_df[recent_df['age_group'] == a]
        sr = (sub_df['predicted_class'] == 1).mean()
        subgroups.append(f"Age {a}")
        selection_rates.append(sr)
        
    # Calculate Disparate Impact relative to privileged classes
    male_sr = selection_rates[1]
    age_ref_sr = selection_rates[4] # Age 36-50
    
    di_ratios = []
    # Gender DI
    di_ratios.append(selection_rates[0] / male_sr if male_sr > 0 else 1.0) # Female DI
    di_ratios.append(1.0) # Male reference
    # Age DI
    for i in range(2, 6):
        di_ratios.append(selection_rates[i] / age_ref_sr if age_ref_sr > 0 else 1.0)
        
    groups_labels = ['Female', 'Male (Ref)', 'Age 18-25', 'Age 26-35', 'Age 36-50 (Ref)', 'Age 50+']
    
    colors_di = ['#FB7185' if x < 0.8 or x > 1.25 else '#34D399' for x in di_ratios]
    colors_di[1] = '#94A3B8' # Reference groups
    colors_di[4] = '#94A3B8'
    
    bars = ax_fairness.bar(groups_labels, di_ratios, color=colors_di, edgecolor='#475569', width=0.6)
    
    # Add guidelines for 80% rule
    ax_fairness.axhline(y=0.80, color='#EF4444', linestyle=':', linewidth=1.5, label='Fairness Lower Bound (0.80)')
    ax_fairness.axhline(y=1.25, color='#EF4444', linestyle=':', linewidth=1.5, label='Fairness Upper Bound (1.25)')
    
    ax_fairness.set_title('B. Disparate Impact Ratio by Subgroup (Last 15 Days)', fontsize=14, pad=12, weight='bold', color='#F8FAFC')
    ax_fairness.set_ylabel('Disparate Impact Ratio', fontsize=11)
    ax_fairness.set_ylim(0, 1.5)
    ax_fairness.grid(True, axis='y', ls=":", color='#334155', alpha=0.5)
    
    # Annotate bars with values
    for bar in bars:
        yval = bar.get_height()
        ax_fairness.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{yval:.2f}", 
                         ha='center', va='bottom', fontsize=10, color='#F8FAFC', weight='bold')
    
    ax_fairness.legend(facecolor='#1E293B', edgecolor='#475569', labelcolor='#F8FAFC', loc='upper right')
    
    # --- PANEL C: Subgroup ROC-AUC comparison ---
    ax_roc = fig.add_subplot(gs[1, 0])
    ax_roc.set_facecolor('#1E293B')
    
    # Calculate ROC-AUC scores for each group (in the whole dataset)
    # We ensure we only compute ROC-AUC for cohorts with at least one positive and one negative sample
    cohorts = [
        ('Overall', df_logs),
        ('Male', df_logs[df_logs['gender'] == 'Male']),
        ('Female', df_logs[df_logs['gender'] == 'Female']),
        ('Age 18-25', df_logs[df_logs['age_group'] == '18-25']),
        ('Age 26-35', df_logs[df_logs['age_group'] == '26-35']),
        ('Age 36-50', df_logs[df_logs['age_group'] == '36-50']),
        ('Age 50+', df_logs[df_logs['age_group'] == '50+'])
    ]
    
    cohort_names = []
    auc_scores = []
    
    for name, cohort_df in cohorts:
        if len(cohort_df) > 0 and len(cohort_df['actual_label'].unique()) > 1:
            auc = roc_auc_score(cohort_df['actual_label'], cohort_df['raw_score'])
            cohort_names.append(name)
            auc_scores.append(auc)
            
    # Horizontal bar plot
    y_pos = np.arange(len(cohort_names))
    bars_roc = ax_roc.barh(y_pos, auc_scores, color='#60A5FA', edgecolor='#475569', height=0.5)
    
    ax_roc.set_yticks(y_pos)
    ax_roc.set_yticklabels(cohort_names, fontsize=11, color='#F8FAFC')
    ax_roc.invert_yaxis()  # top-down list
    ax_roc.set_xlabel('ROC-AUC Score', fontsize=11)
    ax_roc.set_xlim(0.5, 1.0)
    ax_roc.set_title('C. Subgroup ROC-AUC Performance Comparison', fontsize=14, pad=12, weight='bold', color='#F8FAFC')
    ax_roc.grid(True, axis='x', ls=":", color='#334155', alpha=0.5)
    
    # Annotate ROC scores
    for bar in bars_roc:
        xval = bar.get_width()
        ax_roc.text(xval + 0.01, bar.get_y() + bar.get_height()/2.0, f"{xval:.3f}", 
                    ha='left', va='center', fontsize=10, color='#F8FAFC', weight='bold')
        
    # --- PANEL D: KPI Scorecard & Summary Info ---
    ax_kpi = fig.add_subplot(gs[1, 1])
    ax_kpi.axis('off')
    
    # Calculate KPI parameters
    last_audit = df_audit.iloc[-1]
    first_audit = df_audit.iloc[0]
    
    # Color indicators
    fairness_status = "FAIL" if (last_audit['disparate_impact_ratio'] < 0.8 or last_audit['disparate_impact_ratio'] > 1.25) else "PASS"
    drift_status = "DETECTED" if last_audit['drift_p_value'] < 0.05 else "STABLE"
    
    kpi_text = (
        "                    AI Governance KPI Scorecard\n"
        "                    ---------------------------\n\n"
        f"  • Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  • Total Logs Audited: {len(df_logs):,} records (60 Days)\n"
        f"  • Daily Ingestion Vol: 500 inferences/day\n"
        f"  • Feature Drift Status: {drift_status}\n"
        f"    - Baseline p-value: {first_audit['drift_p_value']:.4f}\n"
        f"    - Current p-value: {last_audit['drift_p_value']:.2e}\n"
        f"    - Current PSI: {last_audit['psi_score']:.3f} "
        f"({'major shift' if last_audit['psi_score'] > 0.25 else 'moderate shift' if last_audit['psi_score'] > 0.1 else 'no significant shift'})\n\n"
        f"  • Demographic Parity Status: {fairness_status}\n"
        f"    - Disparate Impact Ratio (Current): {last_audit['disparate_impact_ratio']:.3f}\n"
        f"    - Demographic Parity Ratio (Current): {last_audit['demographic_parity_ratio']:.3f}\n\n"
        f"  • Model Performance Degradation:\n"
        f"    - Accuracy Decay: {last_audit['accuracy_degradation']*100:.2f}% drop from baseline\n"
    )
    
    # Draw text box
    ax_kpi.text(0.05, 0.90, kpi_text, fontsize=12, family='monospace', color='#F8FAFC',
                va='top', bbox=dict(boxstyle='round,pad=1.2', facecolor='#1E293B', edgecolor='#475569'))
    
    # Add EU AI Act Compliance Banner
    ax_kpi.text(0.05, 0.12, 
                "⚠️ COMPLIANCE ALERT: Under the EU AI Act (High-Risk Systems),\n"
                "   disparate impact scores violating the 80% rule require\n"
                "   mandatory mitigation and logging. Feature drift indicates\n"
                "   model calibration mismatch. Immediate retraining is advised.",
                fontsize=11.5, color='#FCA5A5', weight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#7F1D1D', edgecolor='#EF4444'))
    
    # Save the output image
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, facecolor='#0F172A', edgecolor='none', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Dashboard visualization successfully saved to: {output_image_path}")

if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_file = os.path.join(project_dir, 'data', 'observability.db')
    out_file = os.path.join(project_dir, 'viz', 'tableau_ai_observability.png')
    
    generate_governance_dashboard(db_file, out_file)
