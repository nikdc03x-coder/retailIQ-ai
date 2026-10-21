# Phase 02 — Data Quality and Transformation

**Project:** RetailIQ AI  
**Branch:** `phase-02-data-quality`  
**Status:** Transformation validated; automated testing pending

## 1. Research Objective

Investigate the reliability of the Walmart historical sales dataset and create a reproducible analytical dataset suitable for business analytics and future forecasting experiments.

## 2. Methods

We used Pandas for key validation, missing-value analysis, sales anomaly investigation and relational joins.

We used Parquet for efficient storage of the transformed dataset.

The original CSV files were not modified.

## 3. Key Findings

### Dataset relationships

- Sales records: 421,570
- Duplicate business keys: 0
- Missing business keys: 0
- Unmatched feature records for sales: 0
- Sales records without store metadata: 0
- Holiday-flag mismatches: 0

### Missing values

Promotional markdown features contain substantial missing data. Across matched training records, missingness ranges from 64.08% to 73.61%.

Missingness also varies by year, indicating a time-related pattern of feature availability.

CPI and unemployment have no missing values in the matched historical training records.

### Sales anomalies

- Negative weekly sales records: 1,285
- Zero weekly sales records: 73
- Minimum weekly sales: -4,988.94
- Maximum weekly sales: 693,099.36
- Sum of negative sales values: -88,161.56

Negative values appear throughout 2010–2012 and are concentrated in some departments. Their underlying business cause remains unknown.

## 4. Transformation Decisions

1. Preserve all original sales records and values.
2. Preserve negative and zero sales rather than deleting them without evidence.
3. Preserve missing promotional values.
4. Create missingness indicators for each MarkDown feature.
5. Join features using Store and Date.
6. Join store metadata using Store.
7. Reject duplicate or missing business keys.
8. Verify agreement between holiday indicators.
9. Add calendar fields derived from Date.
10. Add an Is_Negative_Sales flag for descriptive analysis only.

The negative-sales flag must not be included in forecasting model inputs because it is derived from the target variable.

## 5. Analytical Dataset

**Output:** `data/processed/retail_analytical.parquet`

- Rows: 421,570
- Columns: 25
- Date range: 2010-02-05 to 2012-10-26
- Approximate file size: 3.3 MB

## 6. Validation Results

The initial independent validation confirmed:

- Row count preserved
- Original sales values preserved
- No duplicate business keys
- Negative sales preserved
- Store metadata complete
- MarkDown missingness indicators correct

The transformation also checks source keys, feature matches and holiday consistency before producing its output.

## 7. Limitations

- The dataset is historical and does not represent current retail operations.
- Negative sales cannot be conclusively identified as returns.
- Missing promotional values cannot automatically be interpreted as zero.
- We have not yet selected or evaluated an imputation strategy.
- We have not yet performed forecasting model evaluation.
- Calendar and target-derived fields must be reviewed for leakage before model training.

## 8. Next Phase

Phase 03 will focus on descriptive business analytics, including historical sales trends, store and department performance, and holiday-related patterns.

Forecasting will follow in a separate phase with chronological validation and baseline model comparisons.
