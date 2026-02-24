# analytics/retrain_response.py
"""Measure what acting on the drift alert is worth: retrain the model every day after the first KS drift alert, on the most recent
week of outcomes that have arrived (labels lag predictions by 7 days), and compare the accumulated accuracy loss against leaving the
deployed model in place. Writes exports/retrain_response.csv.
"""

import os
import sqlite3

import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURES = ["input_feature_1", "input_feature_2"]
LABEL_LAG_DAYS = 7
TRAIN_WINDOW_DAYS = 7
BASELINE_DAYS = 5


def accuracy_loss(df: pd.DataFrame, first_alert_day: int, lag: int = LABEL_LAG_DAYS, window: int = TRAIN_WINDOW_DAYS) -> dict:
    """`df` holds inference logs with a 0-based `day` index. Returns the summed daily accuracy shortfall against the baseline
    days for the deployed model and for daily retraining from `first_alert_day` on."""
    base = df[df["day"] < BASELINE_DAYS]
    baseline = (base["predicted_class"] == base["actual_label"]).mean()
    model, static, retrained = None, [], []
    for day in sorted(df["day"].unique()):
        today = df[df["day"] == day]
        if day >= first_alert_day:
            known = df[(df["day"] <= day - lag) & (df["day"] > day - lag - window)]
            if known["actual_label"].nunique() == 2:
                model = LogisticRegression(max_iter=1000).fit(known[FEATURES], known["actual_label"])
        predicted = model.predict(today[FEATURES]) if model is not None else today["predicted_class"]
        static.append((today["predicted_class"] == today["actual_label"]).mean())
        retrained.append((predicted == today["actual_label"]).mean())
    loss_static = sum(max(baseline - a, 0.0) for a in static)
    loss_retrained = sum(max(baseline - a, 0.0) for a in retrained)
    return {"baseline_accuracy": round(baseline, 4), "first_alert_day": int(first_alert_day) + 1,
            "loss_without_response": round(loss_static, 4), "loss_with_retraining": round(loss_retrained, 4),
            "loss_reduction": round(1 - loss_retrained / loss_static, 4) if loss_static else 0.0,
            "final_accuracy_without": round(static[-1], 4), "final_accuracy_with": round(retrained[-1], 4)}


def main(db_path):
    conn = sqlite3.connect(db_path)
    logs = pd.read_sql_query("SELECT * FROM inference_logs", conn)
    alerts = pd.read_sql_query("SELECT audit_date FROM audit_alerts WHERE rule = 'ks_drift' ORDER BY audit_date", conn)
    conn.close()
    dates = pd.to_datetime(logs["prediction_timestamp"]).dt.strftime("%Y-%m-%d")
    order = {d: i for i, d in enumerate(sorted(dates.unique()))}
    logs["day"] = dates.map(order)
    result = accuracy_loss(logs, order[alerts["audit_date"].iloc[0]])
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame([result]).to_csv(os.path.join(out, "retrain_response.csv"), index=False)
    print(f"Retraining after the drift alert cuts the accumulated accuracy loss by {result['loss_reduction']:.1%} "
          f"(final-day accuracy {result['final_accuracy_with']:.1%} against {result['final_accuracy_without']:.1%})")
    return result


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(os.path.join(project_dir, 'data', 'observability.db'))
