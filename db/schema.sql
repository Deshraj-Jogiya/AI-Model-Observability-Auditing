-- db/schema.sql
-- SQLite Schema for AI Model Observability & Fairness Auditing

-- Drop tables if they exist to ensure clean setup
DROP TABLE IF EXISTS inference_logs;
DROP TABLE IF EXISTS fairness_audit_history;

-- Table to store individual model inference transactions
CREATE TABLE inference_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_timestamp TIMESTAMP NOT NULL,
    gender TEXT NOT NULL,
    age_group TEXT NOT NULL,
    raw_score REAL NOT NULL,          -- Model output probability (e.g., loan approval probability)
    predicted_class INTEGER NOT NULL,  -- Binary decision (1 = Approved, 0 = Denied)
    actual_label INTEGER,             -- Ground truth outcome (1 = Defaulted/Repaid, etc.), can be NULL initially
    input_feature_1 REAL NOT NULL,    -- Continuous input variable 1 (e.g., Debt-to-Income Ratio)
    input_feature_2 REAL NOT NULL     -- Continuous input variable 2 (e.g., FICO Score / Credit History Length)
);

-- Table to store daily or periodic fairness and drift audit metrics
CREATE TABLE fairness_audit_history (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_date TEXT NOT NULL UNIQUE,
    disparate_impact_ratio REAL NOT NULL,
    demographic_parity_ratio REAL NOT NULL,
    drift_p_value REAL NOT NULL,      -- Kolmogorov-Smirnov test p-value for feature drift
    accuracy_degradation REAL NOT NULL -- Performance decay relative to baseline accuracy
);

-- Create indexes for performance tuning on common queries
CREATE INDEX idx_inference_logs_timestamp ON inference_logs(prediction_timestamp);
CREATE INDEX idx_inference_logs_demographics ON inference_logs(gender, age_group);
CREATE INDEX idx_inference_logs_actual_label ON inference_logs(actual_label);
