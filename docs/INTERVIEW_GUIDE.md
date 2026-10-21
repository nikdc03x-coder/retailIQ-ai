# RetailIQ AI: interview and presentation guide

## A 60-second explanation

“I built RetailIQ AI, a Python application that converts historical Walmart retail data into business analytics, evaluated sales forecasts and explainable insights. I validated and joined sales, store metadata and external features while preserving the original sales records. I aggregated department sales into weekly store totals and compared a previous-week baseline, Random Forest and XGBoost using chronological splits and the same evaluation observations. XGBoost reduced held-out MAE by about 25% relative to baseline. I then analyzed reliability, detected unusual residual patterns, and used SHAP to explain predictions. A ten-page Streamlit dashboard presents the saved results. Its business assistant uses pandas for calculations and optional Gemini for constrained question planning and evidence-grounded explanations.”

Use this description only to the extent it accurately represents your own work and understanding. Phase 12 adds a package refactor and operational improvements; earlier experiments already existed.

## How to present it in five minutes

1. **Problem:** retailers need understandable trends, realistic forecasts and a way to investigate unusual sales.
2. **Data:** 421,570 store-department-week records, 45 stores, 143 weeks. Explain the difference between department-level inputs and store-week predictions.
3. **Trustworthy preparation:** enforce unique keys and join cardinality; validate holiday agreement; keep negative sales and missingness indicators.
4. **Forecasting:** previous-week baseline, historical lags and shifted rolling statistics; chronological splits; common evaluation keys; select using validation MAE.
5. **Beyond average accuracy:** per-store WAPE and bias, holiday/week errors, expanding backtests, calibrated anomaly detectors and SHAP reconstruction.
6. **Product layer:** show dashboard filters, a store ranking and an anomaly explanation with original evidence.
7. **Limits and next steps:** historical data, one-week-ahead assumptions, unlabeled anomalies, semantic limits of numeric grounding, and the need for artifact versioning before deployment.

## Engineering questions you should be ready to answer

**Why a feature-based Python package?** The system is a batch analytics pipeline plus a Streamlit UI. Each business feature owns its calculations; common configuration and report access are shared. Package imports work in tests, CLI and dashboard without manipulating `sys.path`. HTTP routes/controllers would add structure without a corresponding server.

**How did you prevent data leakage?** Lag features are shifted before rolling calculations, so a target week's sales cannot enter its predictors. Training precedes validation and test dates. All models reuse split labels. Anomaly thresholds/scaling are calibrated on validation rather than the final test interval. Explain that previous actuals inside later intervals are available under the one-week-ahead assumption.

**Why compare with a naive baseline?** A model must beat an understandable existing strategy to justify its complexity. The baseline uses the previous observed week. Evaluation uses an intersection of store/date/split keys because ML features lose extra warmup weeks.

**Why three metrics?** MAE describes average absolute error in sales units; RMSE emphasizes large misses; WAPE normalizes absolute errors against total absolute sales so stores of different sizes are easier to compare. Bias identifies systematic over/underprediction. Here error is prediction minus actual, so positive bias means overprediction.

**What makes the joins safe?** Sales keys are `(Store, Dept, Date)`, feature keys `(Store, Date)`, and store keys `(Store)`. Many-to-one validation, missing-relationship checks, holiday agreement and row/value preservation prevent silent duplication or lost sales.

**Why not replace every missing promotion with zero?** Missing does not necessarily mean no promotion. The analytical layer preserves unknown values and adds missing flags; the selected store forecasting inputs are lag/calendar features rather than these markdowns.

**How does anomaly detection work?** One detector uses standardized forecast residuals and a validation-calibrated threshold. Isolation Forest uses absolute residual, relative error, weekly change and log sales volume. Both score the same final test period; their union is a review queue, not a labeled fraud detector.

**What does SHAP prove?** It allocates model output relative to a baseline among input features. The project verifies reconstruction against saved predictions. It describes model behavior, not the causal effect of a holiday or promotion on sales.

**What does Gemini actually do?** It may select one allowed operation in structured JSON or explain retrieved evidence. pandas performs numeric calculations. The planner cannot invent arbitrary operations, change ranking direction/count or supply unmentioned store IDs. Explanations with unsupported numbers are withheld.

**Can the assistant still be wrong?** Yes. Matching individual numbers does not verify relationships: a generated answer could associate a valid value with the wrong store or make an unsupported qualitative claim. Evidence stays visible and the documentation avoids claiming complete hallucination prevention.

**How does it work offline?** Analysis, training, dashboard reports and recognized custom sales answers run locally. Without a Gemini key, explanations fall back to source evidence; unfamiliar wording requiring the planner may be unsupported.

**What changed in Phase 12?** Numbered script directories became an installable `src/retailiq` package; manual import hacks were removed; settings, logging, confined CSV access, cache invalidation, API timeouts, ordered pipeline execution, isolated tests, dashboard smoke tests and CI/build configuration were added. The earlier dashboard screens are consolidated into the same dashboard/pages folder and navigation.

## Numbers to remember

| Item | Recorded value |
|---|---:|
| Original sales records | 421,570 |
| Stores / calendar weeks | 45 / 143 |
| Aligned held-out store-week observations | 990 |
| Baseline / RF / XGBoost test MAE | 51,307.24 / 43,642.60 / 38,304.76 |
| XGBoost MAE reduction vs baseline | 25.34% |
| XGBoost overall test WAPE | About 3.68% |
| Statistical / Isolation Forest flags | 10 / 9 |
| Overlap / union | 4 / 15 |
| Expanding-window backtests | 3 folds, 8 weeks each |

These are saved historical results, not universal performance guarantees. Do not describe this as a deployed inventory platform, REST service, database product, real-time pipeline, fraud classifier or multiweek forecasting service.

## A sensible next phase

Add versioned input/model/report manifests and atomic publication first, then reproducible dependency locks and scheduled isolated runs. Validate on newer data and add labeled anomaly outcomes before reporting detection precision. If exposing the dashboard, add authentication and authorization according to the hosting environment. Evaluate multistep forecasting separately rather than assuming one-week metrics transfer.
