# RetailIQ AI — Project Report

**Project:** Retail Sales Analytics, Forecasting and Anomaly Detection  
**Area:** Data Science and Machine Learning  
**Status:** Completed research prototype

## 1. Introduction

Retail businesses collect sales data every week, but rows of numbers do not always make it clear what is happening in the business. A store manager may want to know which stores are doing well, whether sales are changing, and what to expect next week.

I built **RetailIQ AI** to explore how data analysis and machine learning can help answer these questions. I did not want to stop after training one prediction model. I also wanted to check how reliable its predictions were and find unusual sales observations worth investigating.

The project brings these results together in an interactive dashboard.

## 2. Problem and Goal

The main problem is turning historical retail sales data into information that is easier to understand and use.

I focused on four questions:

1. What can we learn from past sales across stores and departments?
2. Can we predict weekly store sales using machine learning?
3. Does the forecasting model perform consistently across stores and time periods?
4. Can we flag unusual observations for further review?

My goal was to build and evaluate a complete data science workflow, not a ready-to-use retail management product.

## 3. Dataset and Preparation

I used a publicly available **Walmart retail sales dataset from Kaggle**. It contains **421,570 sales records** from **45 stores** and **81 departments**, covering **143 weeks from February 2010 to October 2012**.

The data includes weekly sales, store and department information, holiday indicators and related features.

Before the analysis, I checked the structure and quality of the data and prepared it for reuse in later experiments. I kept negative sales values instead of removing them automatically. Such values may reflect returns or adjustments, and dropping them without checking could change the results.

The prepared data was saved in **Parquet** format. This made it possible to reuse the same analytical dataset in different parts of the project.

## 4. Business Analytics

I started by exploring the data before building the forecasting models. I examined sales over time, compared store and department performance, and looked at holiday and non-holiday sales patterns.

This stage helped me understand the scale of the data and how sales differed between parts of the business. I also saved summary reports so the dashboard could display the results without repeating the analysis each time.

One thing I learned here was that understanding the data should come before choosing a machine learning algorithm.

## 5. Sales Forecasting

The forecasting task was to estimate **weekly sales for each store**.

I compared three approaches: a simple **naive baseline**, **Random Forest**, and **XGBoost**. The baseline gave me a reference point. If a more complex model did not perform better than that simple prediction, there would be little reason to prefer it.

The models used information from previous weeks, including lagged sales and rolling averages. I kept the evaluation in time order rather than randomly mixing earlier and later observations.

### Forecasting results

| Model          |      Test MAE |
| -------------- | ------------: |
| Naive baseline |     51,307.24 |
| Random Forest  |     43,642.60 |
| XGBoost        | **38,304.76** |

**XGBoost had the lowest test MAE**, improving on the naive baseline by about **25.34%**.

MAE, or _Mean Absolute Error_, measures the average size of prediction mistakes. A lower MAE is better. These results show how the models performed on the held-out historical test period; they do not guarantee the same performance on future data.

## 6. Forecast Reliability

After selecting XGBoost, I wanted to understand when its forecasts worked well and when they did not.

I compared errors across stores, examined holiday and non-holiday weeks, and used three rolling backtest folds. Rolling backtesting checks performance over different historical evaluation periods rather than depending on only one split.

The main findings were:

- **Test WAPE:** about **3.68%**.
- **Backtest improvements over the baseline:** about **64.81%**, **28.88%**, and **30.41%** across the three folds.
- Errors varied between stores, so one overall score did not tell the full story.
- In the saved holiday comparison, holiday observations had a higher MAE than non-holiday observations.

WAPE, or _Weighted Absolute Percentage Error_, compares the total absolute prediction error with the total actual sales magnitude. It helps express the error relative to the scale of sales.

An important limitation is that this experiment evaluated **one-week-ahead forecasts with previous actual sales available**. It does not demonstrate accurate long-range forecasting without updated sales information.

## 7. Anomaly Detection

I then explored whether the forecasting results could help highlight unusual store-week observations.

I used two methods:

- **Statistical residual analysis:** Looks for unusually large prediction errors using a Z-score threshold.
- **Isolation Forest:** Looks for unusual observations based on the patterns in the features used by the detector.

I compared the two methods because different detectors can flag different observations. Agreement between them is useful to examine, but it does not prove an observation is a real business incident.

### Anomaly detection results

| Result                           | Observations |
| -------------------------------- | -----------: |
| Test observations                |          990 |
| Statistical detector flags       |           10 |
| Isolation Forest flags           |            9 |
| Flagged by both detectors        |            4 |
| Flagged by at least one detector |       **15** |

The detectors had a **Jaccard similarity of about 26.67%** among the observations they flagged. In simple terms, their flagged sets overlapped only partly.

The project has **no verified anomaly labels**, so I cannot report detection accuracy, precision or recall. These 15 observations are **candidates for investigation**, not confirmed fraud, errors or other incidents. The analysis is retrospective, not a live alerting system.

## 8. Interactive Dashboard

To make the findings easier to explore, I built a dashboard with **Streamlit** and **Plotly**.

It has five pages:

| Page                  | What it shows                                                          |
| --------------------- | ---------------------------------------------------------------------- |
| Overview              | Overall historical sales patterns                                      |
| Business Analytics    | Store, department and holiday comparisons                              |
| Sales Forecasting     | Actual versus predicted sales, model comparison and feature importance |
| Forecast Reliability  | Prediction errors, holiday comparisons and rolling backtests           |
| Anomaly Investigation | Detector agreement and filtered observations for review                |

The dashboard reads saved analysis and experiment reports. It does **not** retrain the models when someone opens a page. This keeps the experiments separate from the presentation.

## 9. Testing and Project Workflow

I organized the work into separate Git branches for the main phases, including data preparation, analytics, forecasting, reliability analysis, anomaly detection and the dashboard.

I also wrote automated tests using **pytest**. These checked parts of the data workflow, saved results, forecasting reports, detector counts and dashboard report consistency.

At the end of the dashboard phase, **all 73 automated tests passed**. The same 73 tests passed again after the documentation was merged into `main`.

Passing tests is useful evidence that the tested parts behave as expected, but it is not a guarantee that the project has no errors or that the results will generalize to other datasets.

## 10. What I Learned

The biggest lesson from RetailIQ AI was that a machine learning project is not finished when a model produces a good score.

I learned why it is important to start with data quality checks, compare models against a baseline, keep time order during forecasting evaluation, and investigate where the model makes mistakes.

I also learned to be careful when interpreting anomaly flags. A detector can tell us that something looks unusual, but further evidence is needed to explain why.

Working with Git, tests, saved experiment results and research notes helped me keep the project organized and understand what I had actually built.

## 11. Limitations and Future Work

RetailIQ AI is a **research prototype**, not a production retail system.

Its main limitations are:

- The dataset covers **2010–2012**, so the patterns may not reflect current retail behavior.
- Forecasting results come from historical tests and may change with new data.
- The one-week-ahead evaluation uses previous actual sales; longer forecasting horizons need separate experiments.
- Holiday and store comparisons are descriptive and do not establish the causes of differences.
- The anomaly flags cannot be verified without confirmed labels or a real investigation process.
- The dashboard uses saved results rather than a live retail database.

Possible future improvements include testing with newer datasets, evaluating longer forecast horizons, collecting verified anomaly examples and connecting the workflow to an updated sales database.

## 12. Conclusion

RetailIQ AI helped me put several parts of data science into one project: understanding data, comparing forecasting models, checking prediction reliability, investigating unusual observations and presenting the results clearly.

XGBoost performed best in the saved forecasting comparison, but the reliability and anomaly experiments showed why a single performance score is not enough to judge a system.

For me, the value of this project is both in the results and in learning how to build, test and explain a complete machine learning workflow for a practical business problem.

---

_This report describes results from historical experiments. It does not claim proven performance on future business data._
