# Phase 06 — Retail Sales Anomaly Detection

**Project:** RetailIQ AI  
**Phase:** 06 — Anomaly Detection  
**Status:** Final automated verification completed successfully: 14 Phase 06 tests passed, and all 59 project tests passed.

## 1. Research Objective

Retail businesses need a practical way to identify unexpected changes in store-level sales. Manually reviewing every store and every week is time-consuming.

This phase investigates whether statistical and machine-learning methods can identify unusual store-week sales behavior using the forecasts developed in Phase 04.

The objective is to produce a reproducible investigation queue, not to claim that every flagged observation represents a genuine business incident.

## 2. Research Questions

1. Can standardized forecasting residuals identify unusually large sales deviations?
2. Can Isolation Forest detect unusual multivariate patterns that a global statistical threshold misses?
3. How much do the two detectors agree?
4. Which stores have repeated observations requiring investigation?

## 3. Data and Experimental Setup

The experiments use saved XGBoost predictions from Phase 04 and historical features from the sales forecasting pipeline.

The analysis covers 45 retail stores.

- Calibration observations: 945, from the validation period.
- Held-out test observations: 990, covering 22 weeks.
- Test period: June 1, 2012, through October 26, 2012.
- Observation unit: one store in one week.

The detection methods were fitted or calibrated on validation observations and then applied to the test period.

## 4. Methodology

### 4.1 Statistical residual detector

Residuals were calculated as:

**Residual = Actual sales − Predicted sales**

The mean and standard deviation of validation residuals were used to calculate standardized residuals for test observations.

Observations with an absolute Z-score of at least 3 were flagged.

This method identifies unusually large deviations relative to the global validation residual distribution.

### 4.2 Isolation Forest

Isolation Forest was implemented using scikit-learn.

The experiment used four numerical features:

- Absolute forecasting residual
- Relative forecasting error
- Weekly sales change
- Log-transformed sales volume

StandardScaler was fitted on validation observations only.

Isolation Forest was configured with 200 estimators, contamination of 0.02, and random state 42.

The contamination parameter is a calibration assumption, not a known ground-truth anomaly rate.

### 4.3 Detector comparison

The two methods were compared using their decisions for the same 990 test observations.

Each observation was assigned to one of four categories: both methods, statistical only, Isolation Forest only, or neither.

Jaccard similarity measured overlap among flagged observations. Overall agreement was also calculated.

An investigation queue prioritized observations flagged by both methods for initial review. This is a practical review rule rather than a validated measure of anomaly probability.

## 5. Experimental Results

### 5.1 Statistical detector

The statistical method flagged **10 of 990 observations**, an anomaly rate of **1.01%**.

Validation residual mean: **1,309.13**.

Validation residual standard deviation: **72,238.02**.

The largest absolute standardized residuals included Store 17 on September 7, 2012, and Store 14 on June 22, 2012.

### 5.2 Isolation Forest

Isolation Forest flagged **9 of 990 observations**, an anomaly rate of **0.91%**.

Store 29 was flagged twice by Isolation Forest but was not flagged by the statistical detector.

This suggests that considering relative deviations and other numerical features can produce a different investigation shortlist.

### 5.3 Detector agreement

| Metric                      | Result |
| --------------------------- | -----: |
| Test observations           |    990 |
| Statistical anomalies       |     10 |
| Isolation Forest anomalies  |      9 |
| Flagged by both methods     |      4 |
| Statistical only            |      6 |
| Isolation Forest only       |      5 |
| Flagged by neither          |    975 |
| Unique flagged observations |     15 |
| Jaccard similarity          | 26.67% |
| Overall agreement           | 98.89% |

Although overall agreement is high, the detectors agree on relatively few flagged observations.

Most of the overall agreement comes from observations classified as normal by both methods.

## 6. Store-Level Investigation Findings

Store 14 had four flagged weeks, including three flagged by both methods.

Store 17 had two flagged weeks, including one flagged by both methods.

Store 29 had two flagged weeks, both identified only by Isolation Forest.

Repeated flags may indicate persistent forecasting difficulty or unusual sales behavior. Additional operational information is required before identifying a cause.

## 7. Interpretation

The experiments demonstrate that anomaly-detection methods can produce a compact shortlist for retail investigation.

The two approaches identify overlapping but meaningfully different observations.

Using both detectors provides a broader view than using either method alone, although combining them also increases the number of observations requiring review.

The investigation queue contains 15 unique store-weeks, including four prioritized for initial review because both methods flagged them.

## 8. Limitations

1. The dataset does not provide confirmed anomaly labels. Precision, recall and F1 scores cannot be meaningfully evaluated.
2. Forecasting residuals reflect both unusual sales behavior and forecasting model errors.
3. The statistical detector uses a global threshold and does not explicitly account for store-specific variability.
4. Isolation Forest uses features derived from actual weekly sales, so this is a retrospective detection system rather than an advance warning model.
5. Detector agreement does not establish detection accuracy.
6. The contamination parameter and Z-score threshold were not optimized against labeled anomalies.
7. The dataset is historical and does not establish current retail operating conditions.

## 9. Reproducibility

Implementation scripts:

- `src/retailiq/anomalies/statistical.py`
- `src/retailiq/anomalies/isolation_forest.py`
- `src/retailiq/anomalies/comparison.py`

Generated results are stored under `reports/anomaly_detection/`.

All three experiments were executed successfully.

Manual verification results:

- Statistical detector: 8 checks passed.
- Isolation Forest: 8 checks passed.
- Detector comparison: 10 checks passed.

Final automated pytest verification is pending.

## 10. Conclusion

Phase 06 demonstrates a reproducible approach to detecting unusual store-level sales behavior through statistical analysis and unsupervised machine learning.

The most important finding is that high overall detector agreement can conceal limited agreement on the observations that actually require investigation.

The results support using detector disagreement, repeated store flags and residual magnitude to guide further analysis. They do not establish that either detector is objectively more accurate.

## 11. Next Steps

Complete automated tests, finalize the research log, and merge the verified implementation into the main branch.

Future work may include visualization of flagged observations, store-specific anomaly thresholds and investigation using additional business context.
