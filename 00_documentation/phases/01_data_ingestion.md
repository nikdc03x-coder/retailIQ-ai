# Phase 01 — Dataset Ingestion and Exploration

**Project:** RetailIQ AI — Retail Sales Forecasting and Decision Intelligence  
**Phase:** 01 — Data Ingestion  
**Status:** Automated tests passed; visual verification pending

## 1. Research Objective

The objective of this phase was to acquire, load and inspect a real-world retail sales dataset before performing data cleaning, analytics or machine learning.

The main research question was:

**Is the selected dataset suitable for historical retail sales analysis and forecasting?**

## 2. Dataset Source

- **Dataset:** Walmart Sales Forecast
- **Source:** [Kaggle — aslanahmedov/walmart-sales-forecast](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)
- **Original context:** Historical Walmart store sales forecasting data
- **License displayed by Kaggle:** `copyright-authors`
- **Historical coverage:** February 5, 2010 – October 26, 2012

The downloaded dataset contains `train.csv`, `features.csv`, `stores.csv` and `test.csv`.

For the initial analysis, we imported the first three files. The `test.csv` file remains in the download directory and has not yet been used.

We do not redistribute the original dataset in the Git repository.

## 3. Dataset Structure

| File         | Records | Columns | Purpose                           |
| ------------ | ------: | ------: | --------------------------------- |
| train.csv    | 421,570 |       5 | Historical weekly sales           |
| features.csv |   8,190 |      12 | Promotional and economic features |
| stores.csv   |      45 |       3 | Store characteristics             |

The training dataset contains 45 unique stores, 81 unique department identifiers and 143 unique dates.

**Unit of analysis:** One store–department–week sales observation.

This dataset does not provide individual customer purchase histories.

## 4. Tools and Libraries

- Python 3.12 — Programming environment
- Pandas — CSV loading and data inspection
- Matplotlib — Notebook visualizations
- Plotly — Interactive dashboard visualizations
- Streamlit — Dashboard interface
- Jupyter — Exploratory data analysis
- Pytest — Automated testing
- Git — Version control

We used existing Python libraries instead of implementing data processing or visualization algorithms from scratch.

## 5. Implementation

Created the following components:

- `01_data_ingestion/01_load_csv.py` — Loads the three selected CSV datasets.
- `01_data_ingestion/02_inspect_raw_data.py` — Inspects columns, data types, missing values and exact duplicate rows.
- `14_notebooks/01_dataset_exploration.ipynb` — Explores the dataset and historical weekly sales.
- `11_dashboard/pages/01_Dataset_Overview.py` — Displays dataset information and interactive visualizations.
- `12_tests/test_data_ingestion.py` — Tests the loader and inspection logic.

The inspection script generated `13_results/data_quality/01_raw_data_report.json`.

## 6. Observed Results

### Historical coverage

- Earliest sales date: 2010-02-05
- Latest sales date: 2012-10-26
- Unique recorded weeks: 143
- Invalid training dates: 0

### Missing values

The sales and stores datasets contain no missing cells.

The features dataset contains 24,040 missing cells:

| Feature      | Missing values |
| ------------ | -------------: |
| MarkDown1    |          4,158 |
| MarkDown2    |          5,269 |
| MarkDown3    |          4,577 |
| MarkDown4    |          4,726 |
| MarkDown5    |          4,140 |
| CPI          |            585 |
| Unemployment |            585 |

### Duplicate records

The inspection found zero exact duplicate rows in each of the three imported datasets.

This does not yet establish that their intended business keys are unique.

## 7. Testing and Validation

Automated testing completed successfully:

- Three data-ingestion tests passed.
- Three foundation tests passed.
- Total: **6 passed, 0 failed**.

The tests covered source-file loading, missing-value counting, duplicate detection and required foundation files.

Dashboard rendering and complete notebook execution require separate visual confirmation.

## 8. Business Interpretation

The dataset can support historical analysis of sales patterns across stores, departments and weeks.

Potential investigations include:

- Sales variation over time
- Differences between stores and departments
- Holiday-related sales patterns
- Relationships between recorded features and sales
- Historical sales forecasting

These are research opportunities, not findings established by the ingestion phase.

## 9. Limitations

1. The data ends in 2012 and cannot directly represent present-day Walmart performance.
2. The dataset does not contain individual customer identities or purchase histories.
3. Several promotional and economic features contain missing values.
4. We have not yet validated dataset join keys or temporal consistency.
5. We have not trained or evaluated forecasting models.
6. Dataset reuse and redistribution permissions require careful review.

## 10. Next Steps

Phase 02 will focus on data quality and transformation:

- Validate business keys and date consistency.
- Investigate missing-value patterns.
- Check relationships between source tables.
- Create a reproducible cleaned analytical dataset.
- Record cleaning decisions and their impact.

**Research conclusion:** The dataset is a promising candidate for historical store-level and department-level sales analysis. Further validation is required before forecasting or drawing business conclusions.
