# Phase 04 — Store-Level Sales Forecasting

**Project:** RetailIQ AI  
**Phase:** 04 — Sales Forecasting  
**Status:** Completed — 33 automated tests passed
**Dataset:** Walmart historical weekly sales
**Selected model:** XGBoost

**Selection criterion:** Lowest validation MAE

**Validation MAE:** 48,948.58

**Test MAE:** 38,304.76

**Test RMSE:** 56,584.05

## 1. Research Objective

The objective of this phase is to predict a store's total weekly sales using historical sales patterns and known calendar information.

The central research question is:

**Can machine-learning models improve one-week-ahead retail sales forecasting compared with a simple previous-week baseline?**

## 2. Dataset

The analytical dataset was prepared in Phase 02.

For forecasting, department-level sales were aggregated into store-level weekly totals.

- Stores: 45
- Historical weeks: 143
- Store-week observations: 6,435
- Missing store-weeks: 0
- Observations after feature engineering: 6,255

The historical data covers February 2010 to October 2012.

## 3. Methods

### 3.1 Data preparation

Weekly sales were aggregated by store and date.

An audit verified that every store had observations for all 143 historical weeks.

### 3.2 Feature engineering

The following input features were created:

- Store identifier
- Holiday indicator
- Month, week of year and year
- Sales lagged by 1, 2 and 4 weeks
- Four-week rolling sales mean
- Four-week rolling sales standard deviation

Lag and rolling features were calculated using only historical sales.

The first four weeks per store were excluded because sufficient history was unavailable.

### 3.3 Evaluation strategy

The dataset was divided chronologically.

- Training: through December 30, 2011
- Validation: January 6 to May 25, 2012
- Testing: June 1 to October 26, 2012

Model selection used validation MAE.

The test period was retained for reporting final performance.

### 3.4 Models

Three forecasting approaches were evaluated:

1. Previous-week sales baseline
2. Random Forest Regressor
3. XGBoost Regressor

Existing Python libraries were used rather than implementing the algorithms from scratch.

## 4. Experimental Results

| Model                  | Validation MAE |  Test MAE | Test RMSE |
| ---------------------- | -------------: | --------: | --------: |
| Previous-week baseline |      69,302.24 | 51,307.24 | 77,389.89 |
| Random Forest          |      58,502.77 | 43,642.60 | 63,132.86 |
| XGBoost                |      48,948.58 | 38,304.76 | 56,584.05 |

All models were compared on matching store-week observations.

## 5. Findings

XGBoost achieved the lowest validation MAE and was selected as the preferred model for this experiment.

Its test MAE was approximately 25.34% lower than the previous-week baseline.

Recent sales history was highly influential in both tree-based models, particularly the four-week rolling sales average.

## 6. Interpretation

The findings suggest that recent sales trends and calendar features provide useful predictive information for one-week-ahead store sales forecasting.

However, the results do not establish that XGBoost will perform equally well on new retail datasets or under different economic conditions.

## 7. Limitations

- Only one historical train/validation/test split was evaluated.
- The dataset contains historical observations ending in October 2012.
- Forecasts assume previous-week actual sales are available.
- The experiment forecasts store-level totals, not individual departments.
- The model does not incorporate future unknown promotional information.
- Feature importance does not establish causal relationships.
- Training and validation performance differences require further investigation.
- Additional rolling-origin backtesting would provide stronger evidence of generalization.

## 8. Reproducibility

The experiment uses:

- Python
- Pandas and NumPy
- scikit-learn
- XGBoost
- Joblib
- Pytest

Scripts are stored in `06_sales_forecasting/`.

Generated metrics and predictions are stored in `13_results/sales_forecasting/`.

Trained models are stored in `models/`.

## 9. Conclusion

The experiment demonstrates an end-to-end forecasting workflow, including data auditing, feature engineering, chronological evaluation, baseline comparison and model selection.

XGBoost is the selected model for this historical experiment.

## 10. Next Steps

- Complete automated verification.
- Investigate forecast errors by store and date.
- Consider rolling-origin backtesting.
- Add forecasting results to the Streamlit dashboard.
- Introduce explainability techniques in a later phase.
