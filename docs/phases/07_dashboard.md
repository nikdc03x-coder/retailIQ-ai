# Phase 07 — Interactive Research Dashboard

## 1. Objective

Build an interactive Streamlit dashboard that communicates the results of the RetailIQ AI project to both technical and non-technical users.

The dashboard uses previously generated analytical and machine learning reports rather than retraining models during visualization.

## 2. Research Questions

1. How can historical retail sales patterns be presented to support business decisions?
2. How does XGBoost forecasting performance compare with alternative models?
3. How reliable are forecasts across stores, holiday periods and rolling evaluation windows?
4. How much do statistical residual detection and Isolation Forest agree when identifying unusual sales observations?

## 3. Technology

- Python — application logic
- Streamlit — interactive user interface
- pandas — data loading and transformations
- Plotly — interactive visualizations
- pytest — automated verification
- Parquet/CSV — previously saved analytical data and results

## 4. Dashboard Architecture

The application contains five main pages:

| Page                  | Purpose                                                                           |
| --------------------- | --------------------------------------------------------------------------------- |
| Overview              | Present overall retail sales patterns                                             |
| Business Analytics    | Explore stores, departments and holiday sales                                     |
| Sales Forecasting     | Visualize actual versus predicted sales, model comparisons and feature importance |
| Forecast Reliability  | Explore forecast errors, rolling backtests and holiday reliability                |
| Anomaly Investigation | Compare anomaly detectors and investigate flagged observations                    |

The application loads saved results from the `reports` directory.

## 5. Methods

### Business Analytics

Historical sales are aggregated to study trends across time, stores and departments.

### Sales Forecasting

Saved predictions and evaluation metrics are used to compare forecasting models.

### Forecast Reliability

Store-level errors, holiday error summaries and expanding-window backtests are visualized to identify differences in forecast reliability.

### Anomaly Investigation

Saved statistical residual and Isolation Forest results are compared. Users can filter flagged observations by detector, priority and store.

## 6. Research Findings

### Forecasting

- Selected model: XGBoost
- Test MAE: approximately 38,304.76
- Naive baseline test MAE: approximately 51,307.24
- Relative MAE improvement: approximately 25.34%
- Test WAPE: approximately 3.68%

### Forecast Reliability

- Rolling backtest improvements: approximately 64.81%, 28.88% and 30.41%.
- Holiday observations had a higher reported MAE than non-holiday observations.
- Forecast reliability differs across stores.

### Anomaly Detection

- Test observations: 990
- Statistical detector flags: 10
- Isolation Forest flags: 9
- Flagged by both: 4
- Unique flagged observations: 15
- Jaccard similarity: approximately 26.67%

These flags represent observations requiring investigation, not confirmed business incidents.

## 7. Verification Results

Verification date: 18 October 2026

| Check                            | Result                     |
| -------------------------------- | -------------------------- |
| New dashboard analytics tests    | 8 passed                   |
| Complete automated test suite    | 73 passed                  |
| Python syntax compilation        | Passed                     |
| Git whitespace check             | Passed                     |
| Browser interaction verification | Pending final confirmation |

The new tests validate saved report structures, evaluation metrics, detector count reconciliation and anomaly investigation queue consistency.

## 8. Limitations

1. The dataset represents historical Walmart retail sales from 2010–2012 and may not reflect current retail conditions.
2. The dashboard is based on previously generated reports, not live sales data.
3. The forecasting experiments evaluate one-week-ahead predictions using prior actual sales, not unrestricted multi-week forecasts.
4. Anomaly detection lacks verified ground-truth labels, so detection precision and recall cannot be established.
5. The dashboard provides analytical evidence, but it does not prove that any particular business action will improve sales.

## 9. Conclusion

Phase 07 integrates business analytics, sales forecasting, reliability evaluation and anomaly investigation into a single interactive research dashboard.

The dashboard makes previously generated experimental results accessible for exploration while preserving the limitations of the underlying data and methods.

## 10. Next Steps

- Complete manual browser verification.
- Commit the dashboard implementation and documentation.
- Merge the verified phase branch into `main`.
- Create the Phase 07 release tag.
- Prepare the final project README, research summary and portfolio presentation.
