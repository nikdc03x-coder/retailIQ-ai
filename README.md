# RetailIQ AI

### Understanding Retail Sales Using Data Science and Machine Learning

RetailIQ AI is a data science project that explores how historical retail sales data can help us understand business performance, predict future sales, and identify unusual patterns.

I built this project to learn how different data science techniques can work together to solve a real business problem.

Instead of focusing only on training a machine learning model, I wanted to understand the complete process — from exploring the data to checking whether the predictions are reliable and presenting the results in a dashboard.

## What Problem Does It Solve?

Retail businesses collect large amounts of sales data, but finding useful information in that data is not always easy.

For example, a business might want to know:

- Which stores and departments generate the most sales?
- How do sales change over time?
- Can we predict next week's sales?
- Are the predictions equally reliable for every store?
- Can we identify unusual sales patterns that may need investigation?

RetailIQ AI explores these questions using historical retail data.

## What Can the Project Do?

**1. Business Analytics**

Explore sales trends, compare stores and departments, and understand differences between holiday and non-holiday periods.

**2. Sales Forecasting**

Train and compare machine learning models to predict weekly store sales.

**3. Forecast Reliability**

Measure prediction errors and check how model performance changes across stores and different time periods.

**4. Anomaly Detection**

Use statistical methods and Isolation Forest to flag unusual sales observations for further investigation.

**5. Interactive Dashboard**

Explore the results through a Streamlit dashboard with interactive charts and filters.

## Dataset

This project uses a publicly available Walmart retail sales dataset from Kaggle.

The dataset contains approximately 421,570 sales records from 45 stores, covering February 2010 to October 2012.

It includes weekly sales, store details, department information, holiday indicators and other related features.

The original dataset is used for learning and research. It does not represent live business data.

## Tools Used

| Tool             | Purpose                         |
| ---------------- | ------------------------------- |
| Python           | Main programming language       |
| pandas and NumPy | Data processing and analysis    |
| scikit-learn     | Machine learning and evaluation |
| XGBoost          | Sales forecasting               |
| Plotly           | Interactive charts              |
| Streamlit        | Dashboard                       |
| pytest           | Automated testing               |
| Git              | Version control                 |

## Main Results

The best-performing forecasting model in our selected experiment was XGBoost.

On the held-out test data:

| Metric             |    Result |
| ------------------ | --------: |
| XGBoost MAE        | 38,304.76 |
| Naive baseline MAE | 51,307.24 |
| MAE improvement    |    25.34% |
| XGBoost WAPE       |     3.68% |

The project also compared two anomaly detection approaches. Together, they flagged 15 out of 990 test observations for further review.

These results come from historical experiments. They do not guarantee the same performance on new retail data.

## Run the Dashboard

First, open the project folder and activate the Python virtual environment.

Install the project's required Python libraries using its dependency configuration.

Then start the dashboard:

```bash
python -m streamlit run 08_dashboard/main.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

**Note:** The dashboard reads previously generated analysis and experiment results. Those files must be present before running it.

## Testing

To run the automated tests:

```bash
python -m pytest 12_tests/ -q
```

At the end of Phase 07, all 73 automated tests passed.

## What I Learned

This project helped me understand that building a useful machine learning system involves more than getting a good accuracy score.

I learned why it is important to compare models with simple baselines, test predictions on later time periods, investigate errors and avoid treating every unusual observation as a confirmed anomaly.

I also gained experience organizing a Python project, documenting experiments, writing automated tests and building an interactive dashboard.

## Limitations

RetailIQ AI is a learning and research project, not a production retail management system.

The dataset is historical, forecasting performance may change on new data, and flagged anomalies require human investigation.

The forecasting experiments use one-week-ahead predictions with previous actual sales available.

## Project Documentation

For more details about the project, its experiments and results, see the documentation in the `00_documentation` folder.

The final project report will provide a shorter, complete explanation of the research process.

## Project Status

The data analysis, forecasting, reliability evaluation, anomaly detection and interactive dashboard phases have been implemented.

The project is now in its final documentation and presentation stage.

## Getting Started

This project was developed using Python 3.12.

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd retailiq-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate
```

On Windows PowerShell, activate it using:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install the libraries

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the data

The project uses the Walmart retail sales dataset from Kaggle.

Download the original dataset and follow the project's data preparation and experiment steps.

The dashboard reads saved analytical results from `13_results/`, so those reports must be generated before starting the dashboard.

### 5. Run the dashboard

```bash
python -m streamlit run 08_dashboard/main.py
```

Open the local URL displayed in the terminal.

### 6. Run the tests

```bash
python -m pytest 12_tests/ -q
```
