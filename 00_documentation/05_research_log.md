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
