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
│   ├── dashboard_export.py  # Script to generate a visual governance panel
│   └── tableau_ai_observability.png # Matplotlib visual mockup of the dashboard
├── data/
│   └── observability.db     # SQLite Database containing ingestion & audit tables (gitignored)
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

## 📊 Tableau AI Governance Dashboard Integration

The SQLite output dataset in `data/observability.db` can be directly mapped to **Tableau** to build an interactive, live-updating executive dashboard. 

### 1. Data Connection
1. In Tableau, select **Connect -> To a Server -> More... -> SQLite**.
2. Point Tableau to the local path of `data/observability.db`.
3. Import both tables (`inference_logs` and `fairness_audit_history`) or write a Custom SQL query utilizing the queries provided in [queries.sql](file:///g:/AI-Model-Observability-Auditing/db/queries.sql).

### 2. Suggested Worksheet Layouts in Tableau

* **KPI Summary Tiles**:
  * Set a text block of `AVG(Disparate Impact Ratio)` and color-code using a threshold rule: Red if $< 0.80$, Green if $\ge 0.80$.
  * Set a tile for `MIN(drift_p_value)` for the latest day. Highlight red if $<0.05$.
* **Input Data Drift Trend (Line Chart)**:
  * Place `audit_date` on columns (Continuous) and `drift_p_value` on rows.
  * Right-click the y-axis, select **Logarithmic** scale to highlight small decimal $p$-values.
  * Add a reference line at $y = 0.05$ labeled "Critical Drift Boundary".
* **Subgroup Bias Audits (Bar Charts)**:
  * Rows: `gender` or `age_group`.
  * Columns: Selection Rate (`SUM(predicted_class) / COUNT(inference_logs)`).
  * Use a calculated field `[Unprivileged Selection Rate] / [Privileged Selection Rate]` to plot Disparate Impact directly on a Gantt or Bar chart with reference bounds at 0.80 and 1.25.
* **Accuracy vs Threshold (Dual Axis Chart)**:
  * Track model prediction accuracy vs target outcomes over time to detect calibration drift.

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

Finally, generate the premium dashboard mockup image:
```bash
python viz/dashboard_export.py
```
This saves a professional blue-gray visual report to [viz/tableau_ai_observability.png](file:///g:/AI-Model-Observability-Auditing/viz/tableau_ai_observability.png).

---

## ⚖️ Compliance & Governance Disclaimer
This auditing pipeline is designed to help organizations detect early warning signs of model drift and fairness violations. High-risk systems under the **EU AI Act** are subject to rigorous regulatory criteria, including human oversight, logging, and data governance. Using this pipeline as a component of continuous monitoring helps meet compliance logs requirements but does not substitute for dedicated legal compliance review.
