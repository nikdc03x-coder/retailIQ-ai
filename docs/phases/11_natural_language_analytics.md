# Phase 11 — Natural Language Business Analytics

**Project:** RetailIQ AI — Retail Sales Forecasting & Decision Intelligence  
**Phase:** 11 — Natural Language Business Analytics  
**Status:** Implemented and automated tests passed; final browser verification and release pending  
**Technology:** Python, Pandas, Streamlit, Gemini API, Pytest

---

## 1. Problem Statement

Traditional business dashboards require users to manually navigate charts, select filters, and interpret numerical reports.

Business users may want to ask questions directly, such as:

- Which five stores generated the highest sales?
- Compare Store 10 and Store 20.
- Show the ten lowest-selling departments.

Previously, RetailIQ AI supported a limited collection of predefined business questions. This made it difficult to explore reports using different natural-language expressions.

**Research question:** How can natural-language questions be converted into reliable business analytics while ensuring that numerical answers originate from verified project data?

## 2. Objective

The objective of Phase 11 was to extend the existing Business Assistant with flexible, evidence-based natural-language analytics.

The main goals were:

1. Understand several ways of expressing the same business question.
2. Convert supported questions into approved analytics operations.
3. Perform calculations using Pandas instead of relying on Gemini to calculate sales.
4. Display answers through KPI cards, charts and source records.
5. Reject unsupported requests rather than producing misleading results.
6. Preserve the existing forecasting, anomaly detection and explainability features.

## 3. Dataset and Data Sources

We reused the historical Walmart dataset and the aggregated reports already generated in earlier RetailIQ AI phases.

| Report                 | Records | Important fields       |
| ---------------------- | ------: | ---------------------- |
| Store performance      |      45 | `Store`, `Total_Sales` |
| Department performance |      81 | `Dept`, `Total_Sales`  |

**Report locations:**

- `reports/business_analytics/02_store_performance.csv`
- `reports/business_analytics/03_department_performance.csv`

The source sales dataset contains 421,570 training records covering historical observations from 2010–2012.

The reports provide historical aggregated sales. They do not represent live business transactions.

No new dataset or model training was required for this phase.

## 4. Methodology

We developed a hybrid approach combining rule-based question interpretation, optional Gemini planning, verified Python calculations and evidence-grounded explanations.

### 4.1 Question interpretation

First, the system attempts to recognize the question using regular expressions.

For example:

- "Show top 5 stores by sales."
- "Which five stores generated the most sales?"

Both questions can map to the same operation:

`rank_stores(count=5, highest=True)`

This deterministic method is fast and does not require a Gemini API request.

### 4.2 Gemini-assisted question planning

If a question cannot be recognized by an existing pattern, an optional Gemini planner attempts to interpret it.

The planner can select only approved operations:

- `rank_stores`
- `rank_departments`
- `compare_stores`

The planner returns structured parameters rather than executable code.

Python validates the proposed operation, IDs, counts, direction and requested scope before any calculation.

A question requiring unsupported metrics or periods is not silently converted into an all-time sales question.

### 4.3 Verified data calculations

All custom sales calculations are performed by Pandas functions using saved CSV reports.

The calculations include:

- Highest-selling stores
- Lowest-selling stores
- Highest- and lowest-selling departments
- Comparison of two stores
- Absolute sales differences
- Identification of the higher-selling store

Store rankings support up to 45 stores. Department rankings support up to 81 departments.

### 4.4 Answer generation

The system uses two explanation methods.

**Local Python explanations:** Supported custom sales calculations can be formatted directly from verified evidence, avoiding an additional Gemini explanation request.

**Gemini explanations:** Other supported questions, including forecasting, anomaly detection and SHAP, continue through the existing Gemini service.

Gemini-generated numerical claims are checked against values present in the retrieved evidence before display.

### 4.5 Dashboard visualization

The Streamlit Business Assistant displays relevant results using:

- Natural-language question input with searchable suggestions
- Chat-style conversation history
- KPI cards
- Sales ranking and comparison charts
- Chart and data-table viewing options
- Original source records and report filenames

The interface distinguishes verified local explanations from Gemini-generated explanations.

## 5. System Workflow

```text
User Business Question
        |
        v
Check Question Scope
        |
        v
Existing Pattern Matching
        |
        +---- Recognized ----+
        |                    |
        v                    |
Optional Gemini Planner      |
        |                    |
        v                    |
Validate Operation <---------+
        |
        v
Load Existing CSV Report
        |
        v
Pandas Calculation
        |
        v
Verified Evidence
        |
        v
Local Explanation / Gemini Explanation
        |
        v
Streamlit Answer, KPIs, Charts and Sources
```

Existing forecasting, anomaly and SHAP questions continue through the original Business Context Engine.

## 6. Important Implementation Files

| File                                       | Purpose                                                                     |
| ------------------------------------------ | --------------------------------------------------------------------------- |
| `src/retailiq/dashboard/custom_analytics.py`         | Verified sales ranking and comparison calculations                          |
| `src/retailiq/dashboard/custom_question_router.py`   | Routes supported questions to approved operations                           |
| `src/retailiq/dashboard/question_planner.py`         | Interprets flexible questions using Gemini and validates planned operations |
| `src/retailiq/dashboard/local_answer.py`             | Formats calculated sales results into understandable explanations           |
| `src/retailiq/dashboard/gemini_service.py`           | Handles local answers, grounded Gemini explanations and numerical checking  |
| `src/retailiq/dashboard/business_context.py`         | Preserves previously supported business evidence retrieval                  |
| `src/retailiq/dashboard/pages/business_assistant.py` | Interactive assistant interface, metrics and visualizations                 |

We reused existing Python libraries rather than implementing machine-learning algorithms or user-interface components from scratch.

## 7. Testing and Validation

Automated tests were written using Pytest and controlled sample data.

The test suite checks:

- Correct ascending and descending sales rankings
- Correct store comparisons and sales differences
- Invalid store IDs
- Invalid ranking sizes
- Duplicate records and missing columns
- Unsupported date filters and business metrics
- Incorrect Gemini operation parameters
- Reversal of highest/lowest ranking direction
- Local explanation formatting
- Gemini numerical validation using mock responses
- Compatibility with existing project functionality

**Latest verified test result:**

| Measurement                      | Result                               |
| -------------------------------- | ------------------------------------ |
| Complete automated test suite    | 163 passed                           |
| Test execution time              | 5.29 seconds                         |
| Python syntax checks             | Passed for the main Phase 11 modules |
| Final browser acceptance testing | Not yet confirmed                    |

The passing tests provide evidence of correct behavior for the tested cases. They do not prove that all possible natural-language questions are supported.

## 8. Results and Observations

Phase 11 introduced an additional way to explore historical retail data through natural-language questions.

The project now supports several phrasings of ranking and comparison requests, while preserving the existing Business Assistant features.

The main technical achievement is the separation of responsibilities:

- Question interpretation identifies the intended analysis.
- Validation restricts operations to the approved project scope.
- Pandas calculates numerical results.
- The explanation service presents verified findings.
- Streamlit visualizes the evidence.

This separation reduces reliance on language-model-generated numerical answers.

The implementation was verified through the automated regression suite. Final real-browser behavior remains an acceptance-test requirement.

## 9. Limitations

**Limited supported questions:** The system is not a general-purpose business database question-answering engine. It supports a controlled set of analyses.

**Historical aggregated reports:** Custom sales rankings currently use overall historical totals. Year-specific or monthly calculations are not included.

**No profit analytics:** Profit, costs, margins and quantities are not calculated by these operations.

**Gemini availability:** Questions requiring Gemini planning or explanation depend on the configured API, model access and service availability.

**Numerical grounding limitation:** Existing Gemini validation checks whether numerical values occur in the evidence. It does not fully verify that a number has been attributed to the correct store, department or model.

**Conversation persistence:** Streamlit session-state chat history is session-based rather than a permanent conversation database.

**Evaluation scope:** Automated tests use controlled cases and mocked Gemini responses. Broader user studies and systematic natural-language accuracy benchmarks have not yet been completed.

## 10. Future Improvements

Possible improvements include:

1. A benchmark dataset of natural-language business questions with expected interpretations.
2. Stronger verification of relationships between numerical values and their associated entities.
3. Date-specific analysis using a time-granular sales table.
4. Additional approved analytics operations for growth, trends and comparisons.
5. Improved handling of ambiguous questions through clarification rather than guessing.
6. Performance monitoring for question-planning latency and API usage.

These improvements are outside the current Phase 11 implementation.

## 11. Conclusion

Phase 11 extended RetailIQ AI from a largely predefined-question assistant toward a more flexible Natural Language Business Analytics system.

The approach combines regular expressions, validated Gemini planning, Pandas calculations, local explanations and interactive Streamlit visualizations.

The main learning outcome was understanding how to connect natural-language interfaces with reliable, reproducible business calculations.

**Final status:** Implementation completed and 163 automated tests passed. Browser acceptance testing, documentation commit and final Git release are the remaining release steps.
