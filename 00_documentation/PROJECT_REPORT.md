# RetailIQ AI — Project Report

**Project:** Retail Sales Analytics, Forecasting and Anomaly Detection  
**Area:** Data Science and Machine Learning  
**Status:** Completed prototype

---

## 1. Introduction

Retail stores generate sales data every week. This data can tell us a lot about how a business is performing, but the information is not always easy to understand just by looking at numbers.

For example, a store manager may want to know which departments are performing well, whether sales are increasing or decreasing, or how much stock might be needed in the coming weeks.

I wanted to explore how data science and machine learning could help answer some of these questions.

That is how I started RetailIQ AI.

The idea was to build a project that does more than predict sales. I also wanted to analyze historical business performance, check how reliable the predictions are, and identify unusual sales patterns.

The final result is a Python-based project with an interactive dashboard where users can explore the findings.

## 2. The Problem

Businesses often have access to historical sales data but may struggle to turn it into useful information.

In this project, I focused on four questions:

1. What can we learn about a business from its historical sales?
2. Can machine learning predict weekly store sales?
3. How can we tell whether those predictions are reliable?
4. Can we find unusual sales observations that deserve further investigation?

These questions are connected. A model that predicts sales is useful, but understanding when it makes mistakes is equally important.

Similarly, detecting an unusual observation is only the beginning. We still need to investigate why it happened.

## 3. Dataset and Preparation

I used a publicly available Walmart retail sales dataset from Kaggle.

The dataset contains approximately 421,570 sales records covering 45 stores and 81 departments. The data spans 143 weeks, from February 2010 to October 2012.

It includes weekly sales, holiday information, store characteristics and other related variables.

Before starting the analysis, I checked the dataset structure and prepared the data for further experiments.

One important decision was to preserve negative sales values rather than automatically remove them. Negative sales can represent returns or adjustments, so deleting them without investigation could change the meaning of the data.

The cleaned and prepared data was saved in Parquet format so it could be reused in later phases.

## 4. Business Analytics

I started with business analytics before building any machine learning models.

The goal was to understand the sales data and identify useful patterns.

I explored overall sales trends, store-level performance, department performance and differences between holiday and non-holiday periods.

I also created saved analytical reports that the dashboard could load later.

This phase helped me understand why exploratory data analysis is important. Before predicting future values, we first need to understand what the existing data represents.

## 5. Sales Forecasting

The next step was to predict weekly store sales.

For this experiment, I compared a simple baseline with machine learning approaches, including Random Forest and XGBoost.

The baseline was important because I wanted to know whether a more advanced model was genuinely improving the predictions.

I used past sales information to create features such as previous sales values and rolling averages.

The models were evaluated using time-based data splits rather than randomly mixing historical and future observations.

This helped make the evaluation more realistic for a forecasting problem.

### Forecasting results

| Model          |  Test MAE |
| -------------- | --------: |
| Naive baseline | 51,307.24 |
| Random Forest  | 43,642.60 |
| XGBoost        | 38,304.76 |

XGBoost achieved the lowest test MAE among these models.

Its test MAE was approximately 25.34% lower than the naive baseline.

However, this result does not mean the model will always perform equally well on new data.

## 6. Checking Forecast Reliability

After obtaining a forecasting model, I wanted to know how dependable its predictions were.

A single test score cannot explain whether a model performs equally well across different stores or time periods.

To investigate this, I studied forecasting errors across stores, compared holiday and non-holiday observations, and performed rolling backtesting.

The test WAPE was approximately 3.68%.

The backtesting experiments showed improvements over the baseline of approximately 64.81%, 28.88% and 30.41% across three folds.

I also found that some stores had larger forecasting errors than others.

This showed me that an overall model score can hide important differences in performance.

An important limitation is that the experiments evaluated one-week-ahead forecasts with previous actual sales available. They do not demonstrate unrestricted long-range forecasting.

## 7. Finding Unusual Sales Patterns

For the next phase, I explored anomaly detection.

I wanted to identify observations where sales or prediction errors looked unusual compared with typical patterns.

I used two approaches:

**Statistical residual analysis:** This method examines how unusual the model's prediction errors are compared with a reference distribution.

**Isolation Forest:** This machine learning method looks for observations with unusual combinations of characteristics.

I compared the results of both approaches instead of assuming that one detector was always correct.

### Anomaly detection results

| Result                           | Observations |
| -------------------------------- | -----------: |
| Test observations                |          990 |
| Statistical detector flags       |           10 |
| Isolation Forest flags           |            9 |
| Flagged by both detectors        |            4 |
| Flagged by at least one detector |           15 |

The Jaccard similarity between the sets of flagged observations was approximately 26.67%.

This means the two detectors did not identify exactly the same observations.

An important point is that these are only **potential anomalies**. Because the dataset does not provide verified anomaly labels for this experiment, I cannot claim that every flagged observation represents a real business problem.

## 8. Interactive Dashboard

After completing the experiments, I built a Streamlit dashboard to present the results in a more understandable way.

The dashboard has five pages:

- **Overview:** A summary of historical sales and overall trends.
- **Business Analytics:** Store performance, department performance and holiday comparisons.
- **Sales Forecasting:** Actual versus predicted sales, model performance and feature importance.
- **Forecast Reliability:** Forecast errors, rolling backtests and holiday reliability.
- **Anomaly Investigation:** Detector comparisons, filters and a queue of observations for review.

I used Plotly for interactive charts.

The dashboard loads previously saved reports instead of training the models every time it runs. This keeps visualization separate from experimentation.

## 9. Testing and Verification

I used pytest to check important parts of the project.

The tests covered data processing, analytical results, forecasting outputs, detector comparisons and dashboard report consistency.

At the end of Phase 07, the complete automated test suite passed:

**73 tests passed.**

I also used Git branches and commits to organize development into separate phases.

This helped me track changes and understand which part of the project was developed at each stage.

## 10. What I Learned

This project changed how I think about building machine learning applications.

Initially, it is easy to focus mainly on training a model and improving its accuracy.

But while working on RetailIQ AI, I learned that several other questions are equally important.

For example, I learned why a simple baseline is necessary, why forecasting data should be evaluated in time order, and why we should study model errors instead of looking only at a final performance score.

I also understood that anomaly detection is not the same as confirming that something went wrong.

Another important learning experience was documentation. Writing down the methods, results and limitations helped me understand the project more clearly.

## 11. Limitations and Future Work

RetailIQ AI is a research prototype, not a complete commercial system.

It uses historical sales data, so its findings may not reflect current retail conditions.

The forecasting experiments do not prove that the system can predict many weeks ahead without new actual sales information.

The anomaly detection results also need further validation because confirmed anomaly labels are not available.

In the future, the project could be extended with newer retail datasets, more forecasting experiments, stronger anomaly validation and integration with a real sales database.

## 12. Conclusion

RetailIQ AI helped me explore a complete data science workflow using a real retail dataset.

The project started with understanding historical business data, continued with forecasting and reliability evaluation, and ended with anomaly detection and an interactive dashboard.

The most important outcome for me was not just the final model performance.

It was learning how to connect data analysis, machine learning, evaluation and visualization into a project that addresses a practical business problem.

---

_The results in this report come from the project's saved experiments. They describe historical evaluations rather than guaranteed performance on future data._
