# Phase 03 — Business Analytics and Retail Intelligence

**Project:** RetailIQ AI  
**Branch:** `phase-03-business-analytics`  
**Status:** Analytics implemented; final automated and dashboard verification pending

## 1. Research Objective

Explore historical retail sales data to identify sales patterns, compare store and department performance, and investigate differences between holiday and non-holiday periods.

The objective is descriptive business intelligence, not causal inference or forecasting.

## 2. Dataset

- Historical period: February 2010 to October 2012
- Stores: 45
- Departments: 81
- Weekly periods: 143
- Store–department–week records: 421,570

The analytical dataset was produced and validated during Phase 02.

## 3. Methodology

We used Pandas for data aggregation and statistical summaries.

The analyses include:

1. Weekly sales aggregation across stores and departments.
2. Store-level sales ranking.
3. Department-level sales ranking.
4. Store-week aggregation for holiday comparison.
5. Interactive visualization using Streamlit and Plotly.

All calculations use the original sales values, including negative and zero observations.

## 4. Findings

### 4.1 Weekly sales

Total recorded historical sales: 6,737,218,987.11.

Average weekly sales across all stores: 47,113,419.49.

There are 143 distinct recorded weeks.

### 4.2 Store performance

Store 20 recorded the highest total sales: 301,397,792.46.

The five highest-sales stores are:

- Store 20
- Store 4
- Store 14
- Store 13
- Store 2

All five are classified as Type A.

This association does not establish that store type causes higher sales.

### 4.3 Department performance

Department 92 recorded the highest total sales: 483,943,341.87.

Other high-sales departments include 95, 38, 72 and 90.

These department identifiers do not establish the underlying product categories.

### 4.4 Holiday analysis

Holiday store-week observations: 450.

Non-holiday store-week observations: 5,985.

Average holiday store-week sales: approximately 1.123 million.

Average non-holiday store-week sales: approximately 1.041 million.

The holiday average was approximately 7.84% higher.

The two highest-sales recorded holiday dates were:

- 25 November 2011
- 26 November 2010

This is an observational association and does not establish a causal holiday effect.

## 5. Data Integrity

The independent validation checks confirmed that:

- Weekly sales totals match the original dataset.
- Store and department aggregations preserve sales totals.
- All original records are accounted for.
- Store-week keys are unique.
- Holiday and non-holiday summaries cover all store-week observations.
- Holiday-date totals reconcile with holiday sales.

## 6. Tools and Libraries

- Python
- Pandas
- Plotly
- Streamlit
- Pytest

Existing libraries were used for data manipulation, visualization and testing rather than implementing algorithms from scratch.

## 7. Limitations

1. The data is historical and does not represent current retail performance.
2. Sales are not equivalent to profit.
3. The dataset contains no individual customer transactions.
4. Holiday comparisons may be influenced by seasonality and other factors.
5. Store rankings do not automatically account for store size or local market differences.
6. Negative sales remain in the data because their business meaning is not confirmed.
7. The dashboard supports descriptive analysis, not predictions.

## 8. Next Steps

- Verify the interactive dashboard.
- Complete automated testing.
- Commit and tag the verified Phase 03 work.
- Begin forecasting with chronological train/validation/test splits.
- Compare a simple baseline against machine-learning models.
- Evaluate forecasting performance using MAE and RMSE.
