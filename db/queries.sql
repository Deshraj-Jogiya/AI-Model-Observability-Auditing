-- db/queries.sql
-- Analytical SQL queries for AI Model Observability & Fairness Auditing

-- 1. Tracking Disparate Impact and Fairness Metrics Over Time (Audit History)
-- This query shows historical audits where fairness bounds are breached.
-- The standard regulatory range for Disparate Impact is [0.80, 1.25] (the 80% rule).
SELECT 
    audit_date,
    round(disparate_impact_ratio, 4) AS disparate_impact,
    round(demographic_parity_ratio, 4) AS demographic_parity,
    round(drift_p_value, 6) AS ks_p_value,
    round(accuracy_degradation, 4) AS accuracy_degradation,
    CASE 
        WHEN disparate_impact_ratio < 0.80 OR disparate_impact_ratio > 1.25 THEN 'FAIL: Fairness Breach'
        ELSE 'PASS'
    END AS fairness_status,
    CASE 
        WHEN drift_p_value < 0.05 THEN 'WARN: High Feature Drift'
        ELSE 'PASS'
    END AS drift_status
FROM fairness_audit_history
ORDER BY audit_date DESC;


-- 2. Demographic Subgroup Accuracy and Error Rates
-- Calculates model performance metrics (Accuracy, Selection Rate) grouped by gender and age_group.
-- Only evaluates rows where actual labels (ground truth) are available.
SELECT 
    gender,
    age_group,
    COUNT(*) AS total_cases,
    SUM(predicted_class) AS total_approved,
    ROUND(CAST(SUM(predicted_class) AS REAL) / COUNT(*), 4) AS selection_rate,
    SUM(CASE WHEN predicted_class = actual_label THEN 1 ELSE 0 END) AS correct_predictions,
    ROUND(CAST(SUM(CASE WHEN predicted_class = actual_label THEN 1 ELSE 0 END) AS REAL) / COUNT(*), 4) AS accuracy
FROM inference_logs
WHERE actual_label IS NOT NULL
GROUP BY gender, age_group
ORDER BY gender, age_group;


-- 3. Daily Mean Score and Volatility by Demographic Group
-- Monitors changes in average raw score output and score standard deviation to detect subtle distribution shifts.
SELECT 
    date(prediction_timestamp) AS inference_date,
    gender,
    COUNT(*) AS volume,
    ROUND(AVG(raw_score), 4) AS avg_raw_score,
    -- Standard deviation calculation (SQLite lacks STDDEV, so we use mathematical formula: sqrt(avg(x^2) - avg(x)^2))
    ROUND(SQRT(AVG(raw_score * raw_score) - AVG(raw_score) * AVG(raw_score)), 4) AS std_raw_score
FROM inference_logs
GROUP BY inference_date, gender
ORDER BY inference_date DESC, gender;


-- 4. Disparate Impact Ratio Calculation directly from Raw Inference Logs
-- Compares selection rate of Female (Unprivileged) vs Male (Privileged) subgroup on a rolling weekly basis.
-- Selection Rate = (Approved Females / Total Females) / (Approved Males / Total Males)
WITH WeeklyRates AS (
    SELECT 
        strftime('%Y-%W', prediction_timestamp) AS year_week,
        SUM(CASE WHEN gender = 'Female' AND predicted_class = 1 THEN 1.0 ELSE 0 END) AS female_approved,
        SUM(CASE WHEN gender = 'Female' THEN 1.0 ELSE 0 END) AS female_total,
        SUM(CASE WHEN gender = 'Male' AND predicted_class = 1 THEN 1.0 ELSE 0 END) AS male_approved,
        SUM(CASE WHEN gender = 'Male' THEN 1.0 ELSE 0 END) AS male_total
    FROM inference_logs
    GROUP BY year_week
)
SELECT 
    year_week,
    female_total AS female_volume,
    male_total AS male_volume,
    ROUND(female_approved / female_total, 4) AS female_selection_rate,
    ROUND(male_approved / male_total, 4) AS male_selection_rate,
    ROUND((female_approved / female_total) / (male_approved / male_total), 4) AS calculated_disparate_impact
FROM WeeklyRates
WHERE female_total > 0 AND male_total > 0
ORDER BY year_week DESC;


-- 5. Daily Model Inference Volume and Approval Rates
-- Monitors transaction throughput, ensuring prediction services are running normally.
SELECT 
    date(prediction_timestamp) AS inference_date,
    COUNT(*) AS total_inferences,
    SUM(predicted_class) AS total_approved,
    ROUND(CAST(SUM(predicted_class) AS REAL) / COUNT(*), 4) AS approval_rate
FROM inference_logs
GROUP BY inference_date
ORDER BY inference_date DESC
LIMIT 15;


-- 6. Identifying Subgroups with Disproportionate False Negative Rates (FNR)
-- High False Negative Rates in lending or hiring represent potential fairness issues (Equal Opportunity violation).
-- FNR = False Negatives / (False Negatives + True Positives) = (actual=1, pred=0) / actual=1
SELECT 
    gender,
    age_group,
    SUM(CASE WHEN actual_label = 1 AND predicted_class = 0 THEN 1 ELSE 0 END) AS false_negatives,
    SUM(CASE WHEN actual_label = 1 THEN 1 ELSE 0 END) AS total_actual_positives,
    ROUND(
        CAST(SUM(CASE WHEN actual_label = 1 AND predicted_class = 0 THEN 1 ELSE 0 END) AS REAL) / 
        NULLIF(SUM(CASE WHEN actual_label = 1 THEN 1 ELSE 0 END), 0), 
        4
    ) AS false_negative_rate
FROM inference_logs
WHERE actual_label IS NOT NULL
GROUP BY gender, age_group
ORDER BY false_negative_rate DESC;
