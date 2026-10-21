# Phase 12 path migration inventory

| Old path | New path |
|---|---|
| `00_documentation` | `docs` |
| `01_data_ingestion/01_load_csv.py` | `src/retailiq/data/ingestion.py` |
| `01_data_ingestion/02_inspect_raw_data.py` | `src/retailiq/data/inspection.py` |
| `02_data_quality/01_validate_source_keys.py` | `src/retailiq/data/quality/source_keys.py` |
| `02_data_quality/02_analyze_missing_values.py` | `src/retailiq/data/quality/missing_values.py` |
| `02_data_quality/03_inspect_training_data.py` | `src/retailiq/data/quality/training_data.py` |
| `02_data_quality/04_analyze_sales_anomalies.py` | `src/retailiq/data/quality/sales_anomalies.py` |
| `03_data_transformation/01_build_analytical_dataset.py` | `src/retailiq/data/transformation.py` |
| `05_business_analytics/01_weekly_sales_analysis.py` | `src/retailiq/analytics/weekly.py` |
| `05_business_analytics/02_store_department_analysis.py` | `src/retailiq/analytics/performance.py` |
| `05_business_analytics/03_holiday_sales_analysis.py` | `src/retailiq/analytics/holidays.py` |
| `06_sales_forecasting/01_audit_store_series.py` | `src/retailiq/forecasting/series.py` |
| `06_sales_forecasting/02_baseline_forecast.py` | `src/retailiq/forecasting/baseline.py` |
| `06_sales_forecasting/03_build_features.py` | `src/retailiq/forecasting/features.py` |
| `06_sales_forecasting/04_train_random_forest.py` | `src/retailiq/forecasting/random_forest.py` |
| `06_sales_forecasting/05_train_xgboost.py` | `src/retailiq/forecasting/xgboost_model.py` |
| `06_sales_forecasting/06_compare_models.py` | `src/retailiq/forecasting/comparison.py` |
| `07_anomaly_detection/01_statistical_anomalies.py` | `src/retailiq/anomalies/statistical.py` |
| `07_anomaly_detection/02_isolation_forest.py` | `src/retailiq/anomalies/isolation_forest.py` |
| `07_anomaly_detection/03_compare_detectors.py` | `src/retailiq/anomalies/comparison.py` |
| `07_forecast_reliability/01_analyze_forecast_errors.py` | `src/retailiq/reliability/errors.py` |
| `07_forecast_reliability/02_analyze_relative_errors.py` | `src/retailiq/reliability/relative_errors.py` |
| `07_forecast_reliability/03_rolling_backtest.py` | `src/retailiq/reliability/backtest.py` |
| `08_dashboard/business_context.py` | `src/retailiq/assistant/context.py` |
| `08_dashboard/components` | `src/retailiq/dashboard/components` |
| `08_dashboard/custom_analytics.py` | `src/retailiq/assistant/analytics.py` |
| `08_dashboard/custom_question_router.py` | `src/retailiq/assistant/router.py` |
| `08_dashboard/data_loader.py` | `src/retailiq/dashboard/data_loader.py` |
| `08_dashboard/gemini_service.py` | `src/retailiq/assistant/service.py` |
| `08_dashboard/local_answer.py` | `src/retailiq/assistant/answers.py` |
| `08_dashboard/main.py` | `src/retailiq/dashboard/app.py` |
| `08_dashboard/pages` | `src/retailiq/dashboard/pages` |
| `08_dashboard/question_planner.py` | `src/retailiq/assistant/planner.py` |
| `09_explainable_ai/01_generate_shap_explanations.py` | `src/retailiq/explainability/shap_explanations.py` |
| `11_dashboard/main.py` | `src/retailiq/dashboard/pages/project_overview.py` |
| `11_dashboard/pages/01_Dataset_Overview.py` | `src/retailiq/dashboard/pages/dataset_overview.py` |
| `11_dashboard/pages/02_Business_Analytics.py` | `src/retailiq/dashboard/pages/sales_explorer.py` |
| `12_tests` | `tests` |
| `13_results` | `reports` |
| `14_notebooks` | `notebooks` |
