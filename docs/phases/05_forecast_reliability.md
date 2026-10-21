# Phase 05 — Forecast Reliability Analysis

**Project:** RetailIQ AI  
**Phase:** 05 — Forecast Reliability  
**Status:** Completed and verified — 45 automated tests passed
**Dataset:** Walmart historical weekly sales

## 1. Research Objective

The objective of this phase is to investigate how reliably the selected XGBoost forecasting model performs across stores, weeks and historical evaluation periods.

The research questions are:

1. Which stores and weeks produce the largest forecasting errors?
2. How do forecasting errors compare between stores with different sales volumes?
3. Does XGBoost consistently outperform a simple baseline across multiple historical periods?

## 2. Data and Experimental Context

The analysis uses the forecasting outputs generated during Phase 04.

The held-out test period covers June 1 to October 26, 2012.

- Stores: 45
- Test weeks: 22
- Test predictions: 990
- Selected forecasting model: XGBoost
- Overall test MAE: 38,304.76

No new forecasting models were trained for the store-level error analysis. Previously saved test predictions were reused.

## 3. Methods

### 3.1 Forecast error analysis

For each prediction, signed error and absolute error were calculated.

Signed error was defined as predicted sales minus actual sales.

A positive value indicates overprediction, while a negative value indicates underprediction.

Results were aggregated by store, week and holiday status.

### 3.2 Relative error analysis

Store-level WAPE and bias percentages were calculated to account for differences in sales volume.

WAPE was defined as the sum of absolute forecast errors divided by the sum of absolute actual sales.

This allows stores of different sizes to be compared more meaningfully than with MAE alone.

### 3.3 Rolling-origin backtesting

Three expanding-window historical backtesting folds were evaluated.

Each fold used eight validation weeks and 360 store-week predictions.

XGBoost was retrained using the same fixed hyperparameters as Phase 04.

The original held-out test period was excluded from backtesting.

Predictions represent rolling one-week-ahead forecasting, with observed sales from preceding weeks assumed available.

## 4. Experimental Results

### 4.1 Store-level error analysis

The five stores with the highest test MAE were:

| Store |       MAE |
| ----- | --------: |
| 14    | 87,635.15 |
| 39    | 73,813.35 |
| 28    | 69,638.97 |
| 20    | 69,565.01 |
| 27    | 68,730.54 |

The highest-error week was July 6, 2012, with MAE of 77,738.30.

### 4.2 Holiday comparison

| Week type   | Observations |       MAE |
| ----------- | -----------: | --------: |
| Non-holiday |          945 | 37,030.78 |
| Holiday     |           45 | 65,058.27 |

The holiday comparison contains only one holiday week, so the result is descriptive and cannot establish a general holiday effect.

### 4.3 Relative error analysis

Overall test WAPE was 3.68%.

The five stores with the highest WAPE were:

| Store |  WAPE |
| ----- | ----: |
| 7     | 6.91% |
| 17    | 6.60% |
| 42    | 6.20% |
| 28    | 5.61% |
| 14    | 5.19% |

Store 14 had the highest absolute MAE, but Store 7 had the highest WAPE.

This demonstrates that absolute error and relative error measure different aspects of forecast reliability.

### 4.4 Rolling-origin backtesting

| Fold | Baseline MAE | XGBoost MAE | Improvement |
| ---- | -----------: | ----------: | ----------: |
| 1    |   194,836.78 |   68,564.66 |      64.81% |
| 2    |    54,664.50 |   38,878.54 |      28.88% |
| 3    |    78,760.33 |   54,813.13 |      30.41% |

XGBoost outperformed the baseline in all three folds.

The arithmetic mean of the percentage improvements was 41.36%.

## 5. Key Findings

1. Forecast errors vary meaningfully between stores.
2. Relative error metrics are necessary when comparing stores of different sizes.
3. Some historical weeks produce much larger errors than others.
4. The holiday comparison is limited by the small number of holiday observations.
5. XGBoost outperformed the baseline across all three historical backtesting folds.

## 6. Limitations

- The dataset covers historical retail sales from 2010–2012.
- Store-level reliability was measured over 22 test weeks.
- Only three backtesting folds were evaluated.
- The XGBoost configuration was selected previously, so backtesting results are supporting evidence rather than independent model-selection evidence.
- Rolling one-week-ahead forecasts assume the latest observed sales are available.
- Results do not establish multi-week forecasting performance.
- WAPE can be unstable when actual sales totals are close to zero.
- Observed holiday differences do not establish causation.

## 7. Reproducibility

The experiments use Python, pandas, NumPy, scikit-learn and XGBoost.

Source scripts are stored in `src/retailiq/reliability/`.

Generated reports are stored in `reports/forecast_reliability/`.

Automated verification is implemented using pytest.

## 8. Conclusion

The forecast reliability experiments provide stronger evidence than a single test-set accuracy metric.

XGBoost consistently outperformed the previous-week baseline across three expanding-window historical evaluations.

However, forecasting accuracy varies across stores and calendar weeks. Future work should investigate these differences and evaluate performance on additional datasets.

## 9. Next Steps

- Complete automated verification.
- Document the final experiment status in the research log.
- Integrate forecast reliability insights into the dashboard.
- Explore anomaly detection and explainable AI.
