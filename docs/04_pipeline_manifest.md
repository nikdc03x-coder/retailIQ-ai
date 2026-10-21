# Executable pipeline manifest

Run `python -m retailiq pipeline --list` to inspect the canonical ordered manifest in `src/retailiq/pipeline.py`. Each group below runs its modules in the listed order.

| CLI stage | Modules within `retailiq` | Inputs | Outputs |
|---|---|---|---|
| `data` | `data.inspection`, `data.quality.source_keys`, `data.quality.missing_values`, `data.quality.training_data`, `data.quality.sales_anomalies`, `data.transformation` | `data/raw/{train,features,stores}.csv` | `reports/data_quality/*.json`, `data/processed/retail_analytical.parquet` |
| `analytics` | `analytics.weekly`, `analytics.performance`, `analytics.holidays` | Analytical Parquet | `reports/business_analytics/*.csv` |
| `forecasting` | `forecasting.series`, `forecasting.baseline`, `forecasting.features`, `forecasting.random_forest`, `forecasting.xgboost_model`, `forecasting.comparison` | Analytical Parquet; earlier outputs within group | Store-week series, features, predictions, importances, metrics; RF/XGBoost joblib artifacts |
| `reliability` | `reliability.errors`, `reliability.relative_errors`, `reliability.backtest` | XGBoost predictions, analytical/features datasets | Errors, store/week/holiday summaries, WAPE/bias, rolling backtests |
| `anomalies` | `anomalies.statistical`, `anomalies.isolation_forest`, `anomalies.comparison` | Saved XGBoost predictions and lag features | Scores, flags, agreement, queue and store summary |
| `explainability` | `explainability.shap_explanations` | Trusted XGBoost joblib, original features and predictions | Global SHAP importance, local contributions |

All 22 modules run under `--stage all` (the default). A selected group assumes earlier prerequisites exist. Failure stops subsequent steps. The runner never sends Gemini requests. Existing CSV report filenames retain their research sequence numbers to preserve data contracts and provenance. There is no database stage in the implemented workflow.
