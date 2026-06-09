# HydroPIML-RS: Physics-Informed Adaptive Residual Stacking for Short-Lead Runoff and Stage Forecasting

This repository provides the **computer-code-availability version** of the Python implementation used to support the manuscript:

**Physics informed adaptive residual stacking for short lead joint forecasting of outlet runoff depth and water stage in a debris flow prone small mountain catchment**

The code implements the proposed **HydroPIML-RS** framework for short-lead joint forecasting of outlet runoff depth and outlet water stage in a debris-flow-prone small mountain catchment.

---

## 1. Repository purpose

This repository is intended to support the **Computer code availability** statement of the manuscript.

The main script runs the HydroPIML-RS prediction workflow, including:

- leakage-free hydrological predictor construction;
- chronological train / validation / calibration / independent-test splitting;
- issue-state residual learning;
- hydrological inertia adjustment;
- conservative low-flow treatment for runoff-depth targets;
- adaptive residual stacking;
- nonnegative runoff-depth projection;
- runoff-stage directional consistency processing;
- split conformal uncertainty quantification;
- export of independent-test predictions and performance metrics.

The machine-learning learners listed in the script, including Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, SVR, Ridge, ElasticNet, Huber regression, PLS regression, and gradient boosting, are used to construct the candidate and reference model pool under the same leakage-free chronological protocol.

---

## 2. Repository structure

Main files:

| File or folder | Description |
|---|---|
| `hydropiml_rs_main.py` | Main executable Python script for the HydroPIML-RS prediction experiment. |
| `data.xlsx` | Input daily hydrological dataset. |
| `requirements.txt` | Python package requirements. |
| `README.md` | Repository description and reproducibility instructions. |
| `HydroPIML_RS_code_availability_outputs/` | Output folder automatically generated after running the main script. |
| `predictions/` | Exported prediction CSV files for train, validation, calibration, and independent test subsets. |
| `reference/` | Chronological reference split files. |
| `HydroPIML_RS_prediction_summary.xlsx` | Summary workbook containing metrics, selected candidates, calibration audit, conformal summary, physical-consistency audit, and split diagnostics. |

---

## 3. Input data format

Place the input file in the repository root:

```text
data.xlsx
```

The default worksheet name is:

```text
Daily_Data
```

The expected columns are:

| Column name | Meaning | Unit |
|---|---|---|
| `Date` | Observation date | date |
| `Precipitation_mm` | Daily rainfall | mm d^-1 |
| `Evapotranspiration_mm` | Daily measured evapotranspiration loss | mm d^-1 |
| `Water_level_m` | Outlet water stage | m |
| `Runoff_mm` | Outlet runoff depth | mm d^-1 |

The script automatically converts these columns internally to:

```text
Date, P, ET, H, Q
```

---

## 4. Installation

A clean Python environment is recommended.

Example using `conda`:

```bash
conda create -n hydropiml-rs python=3.10
conda activate hydropiml-rs
pip install -r requirements.txt
```

Example using `venv` on Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, activate the virtual environment using:

```bash
source .venv/bin/activate
```

---

## 5. Running the experiment

After preparing `data.xlsx`, run:

```bash
python hydropiml_rs_main.py
```

The script prints the chronological split, selected HydroPIML-RS configurations, calibration information, physical-consistency diagnostics, conformal uncertainty summary, and independent-test metrics.

The default output folder is:

```text
HydroPIML_RS_code_availability_outputs/
```

---

## 6. Forecasting targets

The implementation evaluates six forecasting targets:

| Horizon | Target | Meaning |
|---:|---|---|
| 1 day | `Q_point` | One-day-ahead outlet runoff depth |
| 1 day | `H_point` | One-day-ahead outlet water stage |
| 3 days | `Q_point` | Three-day-ahead outlet runoff depth |
| 3 days | `H_point` | Three-day-ahead outlet water stage |
| 1--3 days | `Q_max` | Maximum outlet runoff depth within the subsequent three-day forecast window |
| 1--3 days | `H_max` | Maximum outlet water stage within the subsequent three-day forecast window |

The target construction follows the operational rule that all predictors are generated using only information available at or before the forecast issue date.

---

## 7. Leakage-free chronological protocol

The experiment uses a chronological protocol with four subsets:

1. **Training subset**  
   Used to fit candidate learners.

2. **Validation subset**  
   Used for candidate screening, robust ranking, shrinkage selection, stacking selection, and adaptive model-selection decisions.

3. **Calibration subset**  
   Used for post hoc calibration, diagnostic threshold definition, and split conformal uncertainty quantification.

4. **Independent test subset**  
   Used only for final evaluation.

The independent test subset is not used for:

- model fitting;
- candidate selection;
- calibration fitting;
- calibration-method selection;
- ensemble or stacking optimization;
- low-flow or physical-consistency threshold optimization;
- conformal residual quantile estimation;
- final model tuning.

This design supports leakage-free short-lead hydrological forecasting evaluation under seasonal variability, hydrological nonstationarity, and limited high-response samples.

---

## 8. Main output files

After running the script, the following files are generated.

### 8.1 Prediction CSV files

Located in:

```text
HydroPIML_RS_code_availability_outputs/predictions/
```

Example files:

```text
h1_Q_point_predictions.csv
h1_H_point_predictions.csv
h3_Q_point_predictions.csv
h3_H_point_predictions.csv
h3_Q_max_predictions.csv
h3_H_max_predictions.csv
```

Each independent-test prediction file includes:

| Column | Description |
|---|---|
| `issue_date` | Forecast issue date |
| `pred_date` | Target prediction date |
| `Observed` | Observed target value |
| `HydroPIML-RS` | Final HydroPIML-RS prediction |
| `selected_candidate` | Selected target-specific HydroPIML-RS configuration |

Additional train, validation, and calibration prediction files are also exported for audit purposes.

### 8.2 Reference split files

Located in:

```text
HydroPIML_RS_code_availability_outputs/reference/
```

Example files:

```text
h1_train_reference.csv
h1_val_reference.csv
h1_cal_reference.csv
h1_test_reference.csv
h3_train_reference.csv
h3_val_reference.csv
h3_cal_reference.csv
h3_test_reference.csv
```

These files document the chronological subset assignments and observed target values.

### 8.3 Summary workbook

Located at:

```text
HydroPIML_RS_code_availability_outputs/HydroPIML_RS_prediction_summary.xlsx
```

The workbook includes:

| Sheet | Description |
|---|---|
| `run_info` | Overall configuration and reproducibility information |
| `feature_info` | List of generated hydrological predictors |
| `splits` | Chronological split dates and sample counts |
| `split_distribution` | Distribution diagnostics for different subsets |
| `test_metrics` | Independent-test metrics of HydroPIML-RS |
| `selected_candidate` | Selected target-specific HydroPIML-RS configuration |
| `candidate_selection_audit` | Candidate ranking and selection audit |
| `calibration_audit` | Calibration-method audit |
| `conformal_intervals` | Split conformal interval summary |
| `regime_metrics` | Response-regime-specific test metrics |
| `physical_audit` | Runoff-stage physical consistency diagnostics |
| `qh_direction_constraint` | Directional consistency projection audit |
| `stack_weights` | Active stacking weights, if the selected candidate is a stack |

---

## 9. Evaluation metrics

The main independent-test metrics are:

- mean absolute error, MAE;
- root mean square error, RMSE;
- Nash--Sutcliffe efficiency, NSE;
- Kling--Gupta efficiency, KGE;
- mean bias;
- percentage bias, PBIAS;
- peak absolute error.

These metrics are exported in the `test_metrics` sheet of the summary workbook.

---

## 10. Important reproducibility notes

1. The code fixes the random seed using:

   ```python
   RANDOM_STATE = 42
   ```

2. Exact numerical results may vary slightly across operating systems, Python versions, and machine-learning library versions.

3. To reproduce the manuscript results, ensure that:
   - the same quality-controlled `data.xlsx` file is used;
   - the same chronological split policy is used;
   - the same package versions or compatible versions are installed;
   - XGBoost, LightGBM, and CatBoost are available for the full candidate model library.

---

## 11. Computer code availability statement

The Python source code used to implement the HydroPIML-RS framework is available in this repository. The repository includes the main executable script, environment requirements, input-data format description, leakage-free chronological sample construction, hydrological predictor construction, adaptive residual stacking, physical-consistency processing, split conformal uncertainty quantification, and output routines for independent-test predictions and performance metrics.

---

## 12. Contact

**Tianlong Wang**  
Ocean College, Zhejiang University, Zhoushan 316000, China  
School of Civil and Environmental Engineering, Nanyang Technological University, Singapore 637616, Singapore  
Email: <tianlong_wang@zju.edu.cn>
