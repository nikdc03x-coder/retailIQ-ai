# Pipeline Manifest

| Stage | Input | Output |
|---|---|---|
| 01 Ingestion | Raw CSV | Loaded records |
| 02 Quality | Loaded records | Validation report |
| 03 Transformation | Validated records | Clean data |
| 04 Database | Clean data | SQLite tables |
| 05 Analytics | SQLite tables | Business metrics |
| 06 Forecasting | Historical features | Evaluated forecasts |
| 07 Anomalies | Sales history | Unusual-period flags |
| 08 Explainability | Trained model | SHAP explanations |
| 09 Assistant | Verified context | Grounded responses |
| 11 Dashboard | Saved results and APIs | Interactive UI |

Actual script paths and dependencies will be added
when each stage is implemented.
