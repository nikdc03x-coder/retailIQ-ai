# RetailIQ AI

RetailIQ AI turns historical Walmart retail sales into business analytics, evaluated one-week-ahead store forecasts, anomaly investigation queues, SHAP explanations, and evidence-backed business answers in a Streamlit dashboard.

This is a file-based research application. Raw CSVs become a validated Parquet dataset; experiments save CSV reports and joblib model artifacts. The dashboard reads those artifacts without retraining. There is no implemented database, REST API, authentication layer, or live sales integration.

## Features

- Data inspection: keys, duplicates, missing values, source relationships, and unusual raw sales.
- Validated sales/features/store joins that preserve department-level sales records and negative sales.
- Weekly trends, store and department rankings, and holiday comparisons.
- Naive previous-week, Random Forest, and XGBoost forecasts with chronological evaluation.
- MAE, RMSE, per-store WAPE/bias, and three expanding-window backtests.
- Validation-calibrated statistical residual and Isolation Forest detectors with agreement analysis.
- Global and local SHAP contributions, checked against saved predictions.
- Ten pages in one dashboard: overview, dataset overview, business analytics, sales explorer, forecasting, reliability, anomalies, SHAP, assistant and project overview.
- Natural-language rankings and store comparisons calculated in pandas; optional constrained Gemini planning and explanations with numeric checks and offline fallbacks.

## Setup

Use Python 3.12 or later; development and verification used Python 3.12. Run commands from the repository root. On PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS/Linux, activate with `source .venv/bin/activate`. For runtime only use `python -m pip install -r requirements.txt`; for notebooks add `python -m pip install -e ".[notebooks]"`. Runtime dependencies live in `pyproject.toml`.

Optionally copy `.env.example` to `.env`. Leave `GEMINI_API_KEY` blank for local calculations and evidence-only fallback answers. `GEMINI_MODEL`, `GEMINI_TIMEOUT_MS` (100–120000 milliseconds), and `LOG_LEVEL` are validated. Process environment variables take precedence over `.env`. The configured model must be available to your Gemini account.

```powershell
python -m retailiq check-config
```

Never commit `.env`, private datasets, or model artifacts. For a wheel installation or external artifact directory, set `RETAILIQ_HOME` in the process environment to the directory containing `data/`, `reports/`, `models/`, and optionally `.env`. In an editable checkout it defaults to this repository, independently of the working directory.

## Generate reports

Obtain the Walmart Sales Forecast dataset (the original source is recorded in `docs/phases/01_data_ingestion.md`) and put `train.csv`, `features.csv`, and `stores.csv` in `data/raw/`. `test.csv` from the competition contains no actual sales and is not used for the held-out historical evaluation. Raw datasets and generated results are intentionally ignored by Git.

```powershell
python -m retailiq pipeline --list
python -m retailiq pipeline
```

The pipeline runs data inspection/validation/transformation, analytics, forecasting, reliability, anomalies, and SHAP in dependency order. It stops on the first failed stage and writes results to `reports/`, the analytical dataset to `data/processed/`, and fitted models to `models/`. Rerunning overwrites generated artifacts; use `RETAILIQ_HOME` to isolate experiments.

To rerun one group, use `python -m retailiq pipeline --stage analytics` (or `data`, `forecasting`, `reliability`, `anomalies`, `explainability`). Selected groups assume their prerequisite artifacts already exist; they do not automatically regenerate earlier groups. `python scripts/run_pipeline.py --list` is a convenience wrapper. Individual stages run as modules, for example `python -m retailiq.forecasting.features`.

## Run the dashboard

The launcher uses this project's `.venv` and checks that the data needed for all pages is present, readable, nonempty and has the required columns. Source CSVs and analytical Parquet also validate chart inputs such as dates, numeric sales and holiday flags:

```powershell
python start_dashboard.py
```

If port 8501 is already in use, pass `--port 8502`. From another folder, pass the absolute path to `start_dashboard.py`. To check data without starting a server, use `.venv\Scripts\python.exe -m retailiq check-data`.

When using an already activated project environment, you can also run Streamlit directly:

```powershell
python -m streamlit run app.py
```

Use the local URL printed by Streamlit. All pages, including the earlier dataset viewer and sales explorer, share the same sidebar and application configuration. The dashboard needs the generated reports; it does not train models on page visits. Gemini explanations require network access and a key; supported rankings and store comparisons can run entirely locally. Source CSV, report CSV and analytical Parquet changes invalidate dashboard caches using file modification time and size on the next rerun. Dataset Overview and Sales Explorer show recovery instructions for invalid files. Business Analytics supports a single selected store and empty selections.

## Directory structure

```text
app.py                       Streamlit entry point
src/retailiq/
  config.py                  Paths, dotenv loading, typed validated settings
  logging_config.py          Application logging
  reports.py                 Framework-independent confined CSV access
  cli.py, pipeline.py        Ordered research runner
  data/                      Ingestion, quality checks, transformation
  analytics/                 Weekly, store/department, holiday analysis
  forecasting/               Series, baseline, features, models, comparison
  reliability/               Error summaries, relative errors, backtesting
  anomalies/                 Statistical/Isolation Forest scoring, agreement
  explainability/            SHAP generation and reconstruction checks
  assistant/                 Context, analytics, router, planner, service, answers
  dashboard/                 Shared navigation, cached loading and components
    pages/                   All ten dashboard pages with render() functions
tests/                       Existing regressions and new runtime/UI tests
docs/                        Architecture, interview guide, review, phase history
notebooks/                   Learning and exploration notebooks
scripts/                     Thin convenience launchers
data/{raw,processed,validated}/
reports/                     Generated JSON/CSV/Parquet research artifacts
models/                      Generated joblib model artifacts
```

This feature-based Python layout fits the actual application. There are no artificial route/controller folders because the project has no HTTP server. Models in `models/` are serialized ML artifacts; model training code lives in `forecasting/`.

## Verification

```powershell
python -m pytest -q
python -m pytest -m "not integration" -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

The full suite requires local source data and generated reports. `integration` marks these dataset/report-dependent modules and dashboard smoke checks; `-m "not integration"` runs isolated tests on a fresh checkout. Tests disable developer Gemini keys and use fake clients. The GitHub Actions workflow runs isolated tests, lint, formatting, and package builds without private data. A sandbox with a restricted temporary directory may need `--basetemp=<writable-test-directory>`.

## Recorded experiment results

On the aligned 990-row test cohort, XGBoost MAE is **38,304.76**, compared with **51,307.24** for the naive baseline and **43,642.60** for Random Forest: about **25.34%** lower MAE than baseline. XGBoost was selected by validation MAE; test metrics are reported afterward. Overall test WAPE is approximately **3.68%**. The anomaly detectors flag **15 of 990** store-weeks in their union (10 statistical, 9 Isolation Forest, 4 overlap).

These are historical one-week-ahead experiments using previously observed actual sales. They are not recursive multiweek forecasts. Anomaly flags require investigation, and SHAP explains model behavior rather than business causation. Numeric grounding does not prove that every generated statement is factually correct.

## Project guides

- [Architecture and data flow](docs/03_architecture_and_data_flow.md)
- [Executable pipeline manifest](docs/04_pipeline_manifest.md)
- [Interview and presentation guide](docs/INTERVIEW_GUIDE.md)
- [Phase 12 review and migration notes](docs/PHASE_12_REVIEW.md)
- [Historical research report](docs/PROJECT_REPORT.md)

Historical phase documents retain their original experiment/test counts; the Phase 12 review records current verification.
