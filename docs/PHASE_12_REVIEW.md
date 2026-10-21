# Phase 12: code review, restructuring, and verification

Reviewed on October 21, 2026. Branch: `phase-12-feature/refactor-and-structure`.

## Outcome

The existing research and dashboard application has been reorganized into an installable, feature-based Python package. Original application functions/classes are preserved, all imports resolve, and existing research outputs remain reproducible. Changes are local and uncommitted on the requested branch.

The project is a historical retail analytics and ML system: CSV ingestion → validated Parquet → analytics/forecasting/reliability/anomalies/SHAP reports → Streamlit dashboard and constrained business assistant. It uses Python, pandas, NumPy, PyArrow, scikit-learn, XGBoost, joblib, SHAP, Plotly, Streamlit, python-dotenv and the optional Google GenAI SDK. pytest, Ruff and Python build tooling support development.

The restructure makes the code suitable for maintenance and packaging. The application remains a local research system; the changes do not establish a deployed multiuser production service.

## Review findings addressed

| Priority | Finding before review | Change and verification |
|---|---|---|
| P1 | `requirements.txt` contained the malformed combined requirement `python-dotenv>=1,<2matplotlib>=3.8,<4`, duplicated entries, and undeclared direct dependencies such as joblib/PyArrow. | Moved bounded runtime requirements into `pyproject.toml`, separated development/notebook extras, retained requirements-file entry points, installed the package and checked dependency compatibility. |
| P1 | Numbered script names could not be imported conventionally; tests used dynamic file loading and dashboard modules depended on manual `sys.path` changes. | Migrated reusable logic to `src/retailiq`, added package initializers, converted imports and tests to qualified modules, and audited all application module imports inside and outside the checkout. |
| P2 | Gemini requests had no explicit project timeout/retry policy. | Both planner and explanation clients use validated timeout milliseconds and a single SDK attempt. Existing client lifecycle and fallback behavior remain intact. Fake-client regressions verify request settings and cleanup. |
| P2 | Evidence retrieval happened outside the explanation service's exception handling. Missing/corrupt reports could escape as request errors. | Added a stable `reports_unavailable` response, early input validation, and sanitized diagnostic logging. Missing-file, schema and invalid-input regressions exercise these paths. |
| P2 | Dashboard report reads could resolve outside the report directory and report-name-only caching could keep stale data after regeneration. | Added confined framework-independent CSV resolution and cache keys using path/mtime/size. Tests reject traversal and refresh a changed report. Assistant evidence loading no longer imports Streamlit. |
| P2 | Environment parsing was duplicated, with no validation for model, timeout or logging level. | Centralized dotenv/root resolution and immutable settings. Blank API keys remain supported; invalid values produce actionable configuration errors, and secret fields are excluded from representations. |
| P2 | Logging configuration was inconsistent; provider status strings could enter logs. | Added idempotent application logging, stage start/completion/failure events and safe provider-error diagnostics without exception messages or credentials. |
| P2 | A row shift could be called a prior-week lag even when store history skipped calendar weeks. | Baseline and feature generation now require contiguous weekly observations. Gap regressions and historical output comparisons protect the existing complete dataset. |
| P2 | Ranking inputs accepted infinite sales values. | Reject nonfinite/missing sales before ranking or comparison; regressions cover NaN and both infinities. |
| P2 | The pipeline launcher was empty; there was no single executable dependency order. | Added a 22-step CLI runner, stage selection, a nonmutating step list, fail-fast behavior, and an executable manifest. The entire pipeline ran against isolated copied raw data. |
| P2 | Tests could inherit a developer's Gemini key, and there was no clean-checkout CI path or dashboard navigation smoke coverage. | Tests disable inherited API keys and use fake clients. Added integration markers, 43 regression/smoke cases, and a CI workflow for isolated tests, lint, formatting and package builds. |
| P2 | Planning docs described SQLite even though the implementation uses files; the data dictionary still said the dataset was unverified. | Replaced these with implementation-backed architecture, manifest and dictionary; rewrote setup/feature documentation and added an interview guide. Historical phase results are explicitly historical. |

## Why this structure

Features own related batch calculations and UI/service responsibilities have clear boundaries. A `src` layout forces real package imports. Shared modules are limited to paths/settings, logging, report reads and orchestration. Controllers/routes are omitted because the application has no HTTP server. Serialized models remain in `models/`; training code is in `forecasting/`.

All ten dashboard pages are under `dashboard/pages/`. The earlier project overview, dataset explorer and detailed sales explorer now expose `render()` functions and use the same main sidebar and page configuration. Numbering is retained in generated report filenames because those are existing data/provenance contracts, while importable Python filenames use normal names.

## Migration map

| Original location | Current location |
|---|---|
| `00_documentation/` | `docs/` |
| `01_data_ingestion/` | `src/retailiq/data/{ingestion,inspection}.py` |
| `02_data_quality/` | `src/retailiq/data/quality/` |
| `03_data_transformation/` | `src/retailiq/data/transformation.py` |
| `05_business_analytics/` | `src/retailiq/analytics/` |
| `06_sales_forecasting/` | `src/retailiq/forecasting/` |
| `07_forecast_reliability/` | `src/retailiq/reliability/` |
| `07_anomaly_detection/` | `src/retailiq/anomalies/` |
| `08_dashboard/` analytics/assistant helpers | `src/retailiq/assistant/` |
| `08_dashboard/` main/pages/loader/components | `src/retailiq/dashboard/`; root launcher `app.py` |
| `09_explainable_ai/` | `src/retailiq/explainability/` |
| `11_dashboard/` | `src/retailiq/dashboard/pages/{project_overview,dataset_overview,sales_explorer}.py` |
| `12_tests/` | `tests/` |
| `13_results/` | `reports/` (including existing ignored reports) |
| `14_notebooks/` | `notebooks/` |
| `data/`, `models/` | Same artifact locations |

Use `python -m streamlit run app.py` instead of the old dashboard path. Use `python -m pytest` instead of `python -m pytest 12_tests/`. Install with `python -m pip install -e ".[dev]"` before running the new module commands. The exploration notebook uses the shared package configuration and must use a kernel where this package is installed.

## Verification evidence

| Check | Result |
|---|---|
| Existing baseline before migration | 163 passed |
| Full final suite | **206 passed** |
| Isolated suite with `RETAILIQ_HOME` pointing at an empty artifact directory | **92 passed** (isolated suite verified before dashboard consolidation; also included in the current full suite) |
| Dashboard smoke coverage | All 10 destinations in the consolidated dashboard render without runtime exceptions; the three migrated pages also have sidebar-click checks |
| Import migration audit | All **93** previously tracked paths accounted for; every original application-level function/class preserved; no application/test `sys.path` modifications or stale numbered imports |
| Ruff lint and formatting | Pass |
| Dependency consistency | `pip check`: no broken requirements |
| Source distribution and wheel | Build successfully; source package includes launcher/docs/config/test fixtures; every wheel module imports outside checkout |
| Private artifact exclusion | Built packages contain no `.env`, joblib or Parquet artifacts |
| Full isolated pipeline | All **22 stages** succeed |
| Regenerated research artifacts | **35 CSV reports** agree within `rtol=1e-5`, `atol=0.1`; **2 Parquet datasets** and **5 JSON reports** match exactly |
| Git whitespace check | Pass |

The original reports/models were preserved. Pipeline verification wrote to a separate working directory, using copied raw CSVs and `RETAILIQ_HOME`. No live Gemini request was made: SDK integration behavior was exercised with fake clients, and configuration/client construction was checked locally. The GitHub Actions workflow is prepared but has not been executed on GitHub. Tests ran on Windows with Python 3.12; other environments are not yet independently verified.

## Remaining recommendations

| Priority | Evidence / practical limitation | Suggested next step |
|---|---|---|
| P2 | `assistant/service.py::validate_generated_answer` checks numeric membership, not the association between a value and an entity or an unsupported qualitative assertion. | Prefer deterministic templates for more result types, or validate structured claims against specific evidence records before rendering. Add adversarial semantic cases before claiming stronger grounding. |
| P2 | Batch stages publish multiple reports sequentially, while dashboard reads happen independently. Concurrent rebuilds can expose mixed artifact generations. | Add run manifests, versioned output directories and atomic promotion of a completed report set before scheduling shared deployments. Use isolated `RETAILIQ_HOME` directories meanwhile. |
| P2 | Dependency ranges improve compatibility but do not pin a reproducible environment; no data/model lineage manifest exists. | Lock validated environments and record input checksums, package versions, feature schema, parameters and seeds per run. |
| P2 | Schema checks are feature-specific and incomplete for arbitrary replacement datasets; selected store count and supported assistant limits are Walmart-specific. | Introduce typed report contracts/schema versions, fixture-based boundary tests and dataset metadata before generalizing to new retailers. |
| P2 | The current corpus is historical, anomalies have no verified labels and prior actual sales are assumed available weekly. | Validate on newer datasets, label investigation outcomes, measure detection quality, and separately evaluate recursive multistep horizons. |
| P3 | No authentication, authorization, durable chat store, model serving API, scheduler or operational drift monitoring exists. | Add these only for an explicitly scoped hosting/integration phase; they are not necessary for the present local workflow. |

## Interview highlights

Explain the actual engineering: validated merge cardinality, preserved negative/missing sales context, leakage-safe lags, chronological aligned comparisons, baseline-relative gains, reliability beyond one average metric, calibration before anomaly scoring, SHAP reconstruction and bounded LLM involvement. The architecture and interview guide provide the longer narrative and example questions.
