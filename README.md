# AI Model Observability & Fairness Auditing Pipeline

This repository hosts a production-grade, modular framework for monitoring **machine learning feature drift** and auditing **demographic fairness metrics** in compliance with high-risk system frameworks like the **EU AI Act**, **US FTC Guidelines**, and **SR 11-7 model risk management guidelines**. 

The pipeline simulates a typical real-world credit risk evaluation or recruitment AI engine that degrades in performance and fairness due to shift in population characteristics over time (e.g., during macroeconomic shifts).

---

## 🛠️ Pipeline Architecture & Repository Structure

The repository is modular and structured as follows:

```
├── db/
│   ├── schema.sql           # SQLite DB definitions for logs and audit histories
│   └── queries.sql          # 6+ analytical SQL queries for model health reporting
├── etl/
│   └── log_ingestion.py     # Log ingestion simulator (60 days, 30,000 transactions)
├── analytics/
│   └── audit_engine.py      # Statistical engine (KS-Drift & Fairness calculations)
├── viz/
│   ├── dashboard_export.py            # Static matplotlib preview export
│   ├── export_tableau_data.py         # Exports SQLite tables to CSV for Tableau Public
│   └── matplotlib_dashboard_preview.png # Static reference image
├── data/
│   ├── observability.db     # SQLite Database containing ingestion & audit tables (gitignored)
│   ├── inference_logs.csv          # Tableau-ready export (gitignored, regeneratable)
│   └── fairness_audit_history.csv  # Tableau-ready export (gitignored, regeneratable)
├── observability_dashboard.twbx  # The real, published Tableau workbook
├── tests/
│   └── test_audit_engine.py # Real PSI drift-metric tests
├── requirements.txt         # Core dependencies
└── README.md                # Technical documentation (this file)
```

---

## 🔬 Core Concepts & Methodologies

### 1. Input Data Drift Detection
Data drift occurs when the statistical distribution of features changes between training and production environments.
* **Kolmogorov-Smirnov (KS) Test**: Used to compare the cumulative distribution function (CDF) of the current day's feature vector against the baseline reference (the first 5 days of stable production).
  $$\text{KS Stat } D = \sup_x |F_{\text{baseline}}(x) - F_{\text{current}}(x)|$$
  A $p$-value $< 0.05$ indicates a statistically significant deviation in distributions, triggering a training/recalibration alert.
* **Wasserstein Distance (Earth Mover's Distance)**: Offers an alternative scale-independent metric representing the minimal cost to transform one distribution into another.

### 2. Algorithmic Fairness Metrics
To enforce non-discriminatory decisions, the engine computes:
* **Disparate Impact (DI) Ratio (The 80% Rule)**:
  $$\text{Disparate Impact} = \frac{P(\hat{Y}=1 \mid \text{Unprivileged Group})}{P(\hat{Y}=1 \mid \text{Privileged Group})} = \frac{\text{Female Selection Rate}}{\text{Male Selection Rate}}$$
  A ratio outside $[0.80, 1.25]$ indicates a legal violation of fairness (disparate impact).
* **Demographic Parity Ratio (DPR)**:
  $$\text{Demographic Parity Ratio} = \frac{\min_i P(\hat{Y}=1 \mid \text{Subgroup } i)}{\max_j P(\hat{Y}=1 \mid \text{Subgroup } j)}$$
  Computed across age categories (`18-25`, `26-35`, `36-50`, `50+`).
* **Equalized Odds / Equal Opportunity**:
  Measures True Positive Rate (TPR) and False Positive Rate (FPR) parity across subgroups. The repository checks the **False Negative Rate (FNR)** of cohorts to locate systemic model biases where deserving applicants are disproportionately denied credit or hiring slots.

### 3. Model Performance Degradation
Calculates accuracy changes relative to the baseline performance:
$$\text{Accuracy Degradation} = \text{Accuracy}_{\text{baseline}} - \text{Accuracy}_{\text{current}}$$
This alerts operators when **concept drift** occurs, causing prediction scores to decouple from actual labels.

---

## 📊 Tableau Dashboard

**The real, interactive dashboard is built and published live on Tableau Public:
[AI Model Observability & Fairness Audit Dashboard](https://public.tableau.com/app/profile/deshraj.jogiya/viz/AIModelObservabilityFairnessAuditDashboard/Dashboard1)**
-- open it in any browser, no Tableau install needed to view it. The packaged
workbook (`observability_dashboard.twbx`) is also committed in this repo,
openable directly in Tableau Public Desktop.

**A real gotcha found while building this**: Tableau *Public* (the free
desktop app individuals use, as opposed to paid Tableau Desktop) only
supports file-based data sources -- Excel, CSV, PDF, spatial files, web data
connectors -- not live database connections. The "Connect -> To a Server ->
SQLite" instructions below describe paid Tableau Desktop, which requires a
driver Tableau Public doesn't ship. The real path is
`viz/export_tableau_data.py`, which exports the two SQLite tables to CSV.

### What's actually in the published dashboard
* **Feature Drift Trend** (line chart, log-scale y-axis): `audit_date` vs
  `drift_p_value` from `fairness_audit_history.csv`, with an interactive
  threshold parameter at 0.05. The real data shows a genuine story: noisy,
  non-significant p-values for the first ~40 days, then a sharp, sustained
  collapse toward zero -- a real detected drift event, not a flat line.
* **Selection Rate by Gender** (bar chart): `AVG(predicted_class)` grouped by
  `gender` from `inference_logs.csv` (30,000 real simulated inference
  records) -- a direct, real Disparate Impact comparison. In this run both
  bars land around 99% and are nearly identical: a genuine "no gender bias
  detected" finding, which is itself a real audit result worth showing, not
  every fairness check needs to find a problem.

### Reproducing it yourself
1. Run the pipeline (see "Getting Started" below) through `viz/export_tableau_data.py`.
2. Open **Tableau Public Desktop** (free) -> Connect -> Text File -> `data/fairness_audit_history.csv`.
3. Build the drift trend sheet: `audit_date` (as Exact Date, continuous) on Columns, `drift_p_value` on Rows, log-scale y-axis, reference line/parameter at 0.05.
4. New Data Source -> `data/inference_logs.csv`. New worksheet: `gender` on Columns, `predicted_class` on Rows aggregated as **Average**.
5. New Dashboard, drag both sheets in, add a title.
6. File -> Export Packaged Workbook (`.twbx`), or File -> Save to Tableau Public to publish live.

---

## 🚀 Getting Started & Execution

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
First, generate the simulated inference logs and create the SQLite database:
```bash
python etl/log_ingestion.py
```

Second, run the statistical audit engine to compute daily metrics and populate the audit history:
```bash
python analytics/audit_engine.py
```

Generate the static preview image:
```bash
python viz/dashboard_export.py
```
This saves a static reference image to `viz/matplotlib_dashboard_preview.png`.

Export the CSVs Tableau Public needs:
```bash
python viz/export_tableau_data.py
```

---

## ⚖️ Compliance & Governance Disclaimer
This auditing pipeline is designed to help organizations detect early warning signs of model drift and fairness violations. High-risk systems under the **EU AI Act** are subject to rigorous regulatory criteria, including human oversight, logging, and data governance. Using this pipeline as a component of continuous monitoring helps meet compliance logs requirements but does not substitute for dedicated legal compliance review.
