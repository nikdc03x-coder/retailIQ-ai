"""Check the local artifacts needed to populate every dashboard page."""

from pathlib import Path

import pandas as pd

from retailiq.config import ROOT
from retailiq.dashboard.validation import (
    ANALYTICAL_SCHEMA,
    RAW_SCHEMAS,
    require_columns,
    validate_source,
)

REPORT_SCHEMAS = {
    "business_analytics/01_weekly_sales.csv": {
        "Date",
        "Total_Sales",
        "Average_Department_Sales",
        "Sales_Records",
    },
    "business_analytics/02_store_performance.csv": {
        "Store",
        "Type",
        "Total_Sales",
        "Average_Weekly_Sales",
    },
    "business_analytics/03_department_performance.csv": {"Dept", "Total_Sales", "Stores"},
    "business_analytics/04_store_week_sales.csv": {
        "Store",
        "Date",
        "Weekly_Store_Sales",
        "IsHoliday",
    },
    "business_analytics/05_holiday_comparison.csv": {"IsHoliday", "Average_Store_Week_Sales"},
    "business_analytics/06_holiday_dates.csv": {"Date", "Total_Sales", "Stores"},
    "sales_forecasting/10_xgboost_predictions.csv": {
        "Store",
        "Date",
        "Weekly_Store_Sales",
        "Prediction",
        "Split",
    },
    "sales_forecasting/11_xgboost_importance.csv": {"Feature", "Importance"},
    "sales_forecasting/12_aligned_model_comparison.csv": {
        "Model",
        "Split",
        "Observations",
        "MAE",
        "RMSE",
    },
    "forecast_reliability/01_test_prediction_errors.csv": {
        "Store",
        "Date",
        "Weekly_Store_Sales",
        "Prediction",
        "Absolute_Error",
    },
    "forecast_reliability/02_store_error_summary.csv": {
        "Store",
        "Observations",
        "MAE",
        "Mean_Error",
        "Mean_Actual_Sales",
        "RMSE",
    },
    "forecast_reliability/03_week_error_summary.csv": {"Date", "Stores", "MAE", "RMSE"},
    "forecast_reliability/04_holiday_error_summary.csv": {
        "IsHoliday",
        "Observations",
        "MAE",
        "RMSE",
    },
    "forecast_reliability/05_store_relative_errors.csv": {
        "Store",
        "Observations",
        "MAE",
        "WAPE_Pct",
        "Bias_Pct",
    },
    "forecast_reliability/06_rolling_backtest_results.csv": {
        "Fold",
        "Baseline_MAE",
        "XGBoost_MAE",
        "Improvement_Pct",
        "Observations",
    },
    "anomaly_detection/07_detector_comparison.csv": {
        "Store",
        "Date",
        "Weekly_Store_Sales",
        "Prediction",
        "Residual",
        "Z_Score",
        "Statistical_Anomaly",
        "Isolation_Anomaly",
        "Agreement_Category",
        "Any_Anomaly",
        "Review_Priority",
    },
    "anomaly_detection/08_detector_agreement_summary.csv": {
        "Test_Observations",
        "Statistical_Anomalies",
        "Isolation_Anomalies",
        "Both",
        "Statistical_Only",
        "Isolation_Only",
        "Neither",
        "Union",
        "Jaccard_Similarity",
        "Overall_Agreement_Pct",
    },
    "anomaly_detection/09_anomaly_investigation_queue.csv": {
        "Store",
        "Date",
        "Weekly_Store_Sales",
        "Prediction",
        "Residual",
        "Z_Score",
        "Statistical_Anomaly",
        "Isolation_Anomaly",
        "Agreement_Category",
        "Any_Anomaly",
        "Review_Priority",
    },
    "explainable_ai/01_global_shap_importance.csv": {"Feature", "Mean_Absolute_SHAP"},
    "explainable_ai/02_local_shap_contributions.csv": {
        "Store",
        "Date",
        "Weekly_Store_Sales",
        "Prediction",
        "SHAP_Base_Value",
        "SHAP_Reconstructed_Prediction",
        "SHAP_Store",
        "SHAP_IsHoliday",
        "SHAP_Month",
        "SHAP_WeekOfYear",
        "SHAP_Year",
        "SHAP_Lag_1",
        "SHAP_Lag_2",
        "SHAP_Lag_4",
        "SHAP_Rolling_Mean_4",
        "SHAP_Rolling_Std_4",
    },
}

REQUIRED_DATA = (
    "data/raw/train.csv",
    "data/raw/features.csv",
    "data/raw/stores.csv",
    "data/processed/retail_analytical.parquet",
    "reports/business_analytics/01_weekly_sales.csv",
    "reports/business_analytics/02_store_performance.csv",
    "reports/business_analytics/03_department_performance.csv",
    "reports/business_analytics/04_store_week_sales.csv",
    "reports/business_analytics/05_holiday_comparison.csv",
    "reports/business_analytics/06_holiday_dates.csv",
    "reports/sales_forecasting/10_xgboost_predictions.csv",
    "reports/sales_forecasting/11_xgboost_importance.csv",
    "reports/sales_forecasting/12_aligned_model_comparison.csv",
    "reports/forecast_reliability/01_test_prediction_errors.csv",
    "reports/forecast_reliability/02_store_error_summary.csv",
    "reports/forecast_reliability/03_week_error_summary.csv",
    "reports/forecast_reliability/04_holiday_error_summary.csv",
    "reports/forecast_reliability/05_store_relative_errors.csv",
    "reports/forecast_reliability/06_rolling_backtest_results.csv",
    "reports/anomaly_detection/07_detector_comparison.csv",
    "reports/anomaly_detection/08_detector_agreement_summary.csv",
    "reports/anomaly_detection/09_anomaly_investigation_queue.csv",
    "reports/explainable_ai/01_global_shap_importance.csv",
    "reports/explainable_ai/02_local_shap_contributions.csv",
)


def find_data_issues(root: Path = ROOT) -> list[str]:
    """Validate readable artifacts and their dashboard schemas before startup."""
    issues = []
    for relative in REQUIRED_DATA:
        path = root / relative
        try:
            if not path.is_file():
                issues.append(f"Missing: {relative}")
                continue
            if path.stat().st_size == 0:
                issues.append(f"Empty: {relative}")
                continue
            data = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            if relative.startswith("data/raw/"):
                validate_source(data, RAW_SCHEMAS[path.name], relative)
            elif relative == "data/processed/retail_analytical.parquet":
                validate_source(data, ANALYTICAL_SCHEMA, relative)
            else:
                schema = REPORT_SCHEMAS.get(relative.removeprefix("reports/"), set())
                require_columns(data, schema, relative)
        except (OSError, ValueError, TypeError) as error:
            issues.append(f"Invalid: {relative}: {error}")
    return issues
