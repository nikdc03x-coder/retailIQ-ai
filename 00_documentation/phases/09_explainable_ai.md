# Phase 09 — Explainable AI with SHAP

## 1. What Was the Goal?

In the forecasting phase, I trained an XGBoost model to predict weekly store sales. The model gave useful predictions, but the prediction values alone did not explain how the model arrived at them.

In this phase, I wanted to make those predictions easier to understand.

I used SHAP to study how different features influence the model's predictions.

## 2. What Did I Use?

I reused the existing trained XGBoost model and its original forecasting features.

The model uses 10 features, including previous sales values, rolling sales averages, calendar information and store identifiers.

I used the Python SHAP library with TreeExplainer because the forecasting model is based on decision trees.

No new forecasting model was trained.

## 3. How Does It Work?

SHAP calculates a contribution value for each feature in a prediction.

A positive contribution pushes the prediction above the SHAP reference value, while a negative contribution pushes it below.

When we add the contributions to the reference value, we should get approximately the same prediction as the original model.

I created two reports:

- **Global SHAP importance:** Shows which features had the greatest average influence across the test predictions.
- **Local SHAP contributions:** Shows how individual features contributed to a selected store's weekly prediction.

Both reports are generated from the existing XGBoost model.

## 4. Results

I calculated SHAP explanations for 990 test observations covering 45 stores and 10 features.

The five features with the highest average absolute SHAP contributions were:

| Feature        | Mean Absolute SHAP |
| -------------- | -----------------: |
| Rolling_Mean_4 |         246,517.67 |
| Lag_1          |         132,578.38 |
| Lag_4          |          69,076.13 |
| Lag_2          |          42,825.36 |
| WeekOfYear     |          34,920.05 |

The results show that previous sales values and rolling averages have a strong influence on the XGBoost model's predictions.

The maximum difference between a prediction and its reconstruction using SHAP was 2.0 sales units. The average difference was approximately 0.33 sales units.

The numerical reconstruction check passed.

## 5. Dashboard Integration

I added an Explainable AI page to the existing Streamlit dashboard.

The page is designed to show overall feature importance and individual prediction explanations.

Users can choose a store and week to inspect how the model produced a particular forecast.

The page reads saved SHAP results, so the model does not need to be retrained when using the dashboard.

## 6. Limitations

SHAP explains how the trained model uses its features. It does not prove that a particular feature caused a real-world change in sales.

Features such as previous sales and rolling averages are related, so their individual contributions should be interpreted carefully.

The explanations were generated for historical test observations, not for live sales predictions.

## 7. What I Learned

This phase helped me understand the difference between making a prediction and explaining a prediction.

The existing forecasting charts show the predicted values, while SHAP helps explain how those values were calculated.

I also learned why it is important to verify that explanations belong to the exact trained model and reproduce its predictions.

## 8. Next Step

The next phase is to build a Gemini Business Assistant that retrieves verified results from RetailIQ AI and explains them in simple language.

Gemini will use saved analytical results and SHAP evidence rather than inventing new predictions or business facts.
