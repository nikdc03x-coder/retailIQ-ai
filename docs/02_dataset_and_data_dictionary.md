# Dataset and data dictionary

Verified from the local CSV sources and saved data-quality reports. The acquisition source and historical usage notes are recorded in `phases/01_data_ingestion.md`; no raw dataset is redistributed here.

| Source | Rows | Columns | Key / grain |
|---|---:|---:|---|
| `train.csv` | 421,570 | 5 | `(Store, Dept, Date)`: department-store-week sales |
| `features.csv` | 8,190 | 12 | `(Store, Date)`: external weekly features |
| `stores.csv` | 45 | 3 | `(Store)`: store metadata |

Sales cover February 5, 2010 through October 26, 2012: 143 unique weeks, 45 stores, 81 observed department identifiers. Feature data extends beyond the training period; transformation joins only matching sales dates. Competition `test.csv` has no actual sales target and is excluded from the historical model evaluation.

| Field | Meaning / handling |
|---|---|
| `Store`, `Dept` | Integer identifiers; keys must be nonmissing and unique at the source grain. |
| `Date` | Recorded week date; converted to datetime. |
| `Weekly_Sales` | Department weekly target in the original sales records. Legitimate negative values are retained. |
| `IsHoliday` | Boolean holiday flag, required to agree between sales and feature sources. |
| `Temperature`, `Fuel_Price`, `CPI`, `Unemployment` | External weekly numeric context; not in the selected store forecasting feature list. |
| `MarkDown1`–`MarkDown5` | Promotional markdown variables with substantial missingness; preserved with additional missing flags. |
| `Type`, `Size` | Store category and size metadata. |
| `Weekly_Store_Sales` | Derived store-week target: department sales summed within `(Store, Date)`. |
| `Lag_1`, `Lag_2`, `Lag_4` | Prior observed sales at one, two and four contiguous weeks. |
| `Rolling_Mean_4`, `Rolling_Std_4` | Four-week statistics after shifting sales by one week. |

Raw sales have no missing values or exact duplicate rows, but contain 1,285 negative records and 73 zero records. Feature missingness is profiled separately; absence is not assumed to mean a zero promotion. The transformation preserves original rows, order and values and rejects invalid joins. Forecasting aggregation produces 6,435 store-weeks; the four-week feature warmup leaves 6,255 model rows.

The dataset has no customer transactions, inventory availability, costs, profit, confirmed anomaly labels or current retail observations. The assistant's historical rankings are full-period aggregates; requests for unsupported date slices or metrics are rejected.
