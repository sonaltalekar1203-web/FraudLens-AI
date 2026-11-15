
import os
import pandas as pd
import mlflow
from evidently import Report
from evidently.presets import DataDriftPreset

def monitor_fraudlens_drift(current_batch, report_name="drift_report.html"):

    reference_path = os.path.join(
        os.path.dirname(__file__),
        "artifacts",
        "fraudlens_reference_data.parquet"
    )

    reference = pd.read_parquet(reference_path)
    current = current_batch[reference.columns].copy()

    reference = reference.astype("float64")
    current = current.astype("float64")

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference,
        current_data=current
    )

    output_path = os.path.join(
        os.path.dirname(__file__),
        "artifacts",
        report_name
    )

    snapshot.save_html(output_path)

    result = snapshot.dict()["metrics"][0]["value"]
    count = int(result["count"])
    share = float(result["share"])

    return {
        "status": "ALERT" if share >= 0.50 else "NORMAL",
        "drifted_features": count,
        "total_features": len(reference.columns),
        "drift_share": share,
        "report_path": output_path
    }

def monitor_and_log(current_batch, run_name="FraudLens_Automated_Monitoring"):

    result = monitor_fraudlens_drift(current_batch)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics({
            "drifted_features": result["drifted_features"],
            "total_features": result["total_features"],
            "drift_share": result["drift_share"]
        })

        mlflow.set_tags({
            "monitoring_status": result["status"],
            "monitoring_tool": "Evidently",
            "project": "FraudLens AI"
        })

        mlflow.log_artifact(result["report_path"])

    return result
