# Research and Development Log

## Phase 00 — Foundation

### Objective

Create a reproducible Python project with Git,
documentation, notebooks and a working dashboard.

### Tools

Python, Git, VS Code, Streamlit, Pandas and Jupyter.

### Work Completed

Update this section after running the setup.

### Test Results

Record the actual test outputs here.

### Problems Encountered

Record errors and how they were solved.

### Limitations

No real dataset has been analyzed yet.

### Next Phase

Dataset selection, inspection and data ingestion.

## Phase 04 — Sales Forecasting

Date: 2026-10-18

### Objective

Develop and evaluate a one-week-ahead store-level sales
forecasting system.

### Work completed

- Audited historical store-week coverage.
- Verified 45 stores and 143 weeks with no missing store-weeks.
- Created historical lag and rolling sales features.
- Established a previous-week forecasting baseline.
- Trained Random Forest and XGBoost regressors.
- Compared models using chronological validation data.
- Evaluated the selected model on the held-out test period.

### Main findings

- XGBoost achieved the lowest validation MAE: 48,948.58.
- Test MAE: 38,304.76.
- Test RMSE: 56,584.05.
- XGBoost reduced test MAE by approximately 25.34%
  compared with the previous-week baseline.
- Recent sales history was highly influential.

### Verification

- Phase 04 automated tests: 11 passed.
- Entire project test suite: 33 passed.

### Limitations

- One historical chronological split.
- Rolling one-week-ahead forecasting.
- Historical data ends in October 2012.
- Results may not generalize to current retail conditions.

### Next steps

- Analyze prediction errors.
- Consider rolling-origin backtesting.
- Integrate forecasting results into the dashboard.

Status: Experiments completed; final automated verification pending.
Evaluate whether the selected XGBoost model produces reliable forecasts across different stores, weeks and historical evaluation periods.

1. Analyzed 990 test predictions across 45 stores and 22 weeks.
2. Calculated store-level MAE, RMSE, signed errors and holiday comparisons.
3. Calculated store-relative WAPE and bias.
4. Performed three expanding-window backtesting experiments.

- Overall test MAE: 38,304.76.
- Overall test WAPE: 3.68%.
- Highest store-level WAPE: Store 7 at 6.91%.
- Highest store-level MAE: Store 14 at 87,635.15.
- XGBoost outperformed the baseline in all three backtesting folds.
- Backtesting improvements: 64.81%, 28.88% and 30.41%.
  Only one holiday week was present in the held-out test period. The backtesting experiment evaluated rolling one-week-ahead predictions, not forecasts made eight weeks into the future. Additional datasets and evaluation periods would strengthen the findings.
  Complete automated verification, finalize the Phase 05 documentation, and begin anomaly detection.
