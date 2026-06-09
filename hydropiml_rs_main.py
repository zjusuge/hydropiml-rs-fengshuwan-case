# -*- coding: utf-8 -*-
"""
HydroPIML-RS prediction experiment
==================================

This script provides the computer-code-availability version of the
HydroPIML-RS prediction experiment.

Purpose
-------
The script runs the HydroPIML-RS workflow for short-lead joint forecasting of
outlet runoff depth and outlet water stage in a debris-flow-prone small
mountain catchment.

Input
-----
Place data.xlsx in the same directory as this script.

Expected worksheet
------------------
Daily_Data

Expected columns
----------------
Date
Precipitation_mm
Evapotranspiration_mm
Water_level_m
Runoff_mm

Outputs
-------
HydroPIML_RS_code_availability_outputs/
    predictions/
        h1_Q_point_predictions.csv
        h1_H_point_predictions.csv
        h3_Q_point_predictions.csv
        h3_H_point_predictions.csv
        h3_Q_max_predictions.csv
        h3_H_max_predictions.csv
    reference/
        chronological reference split files
    HydroPIML_RS_prediction_summary.xlsx

Scientific rules
----------------
1. Predictors are constructed using information available at or before the
   forecast issue date.
2. Samples are split chronologically into training, validation, calibration,
   and independent testing periods.
3. The independent test period is not used for model fitting, candidate
   selection, calibration fitting, calibration-method selection, ensemble
   optimization, threshold optimization, or Q-H physical-consistency decision.
4. HydroPIML-RS predictions, diagnostics, and performance metrics are reported.
"""

from __future__ import annotations

import os
import re
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression, HuberRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)
from sklearn.isotonic import IsotonicRegression

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


# =============================================================================
# Optional internal learner components
# =============================================================================

OPTIONAL_MODEL_STATUS: Dict[str, str] = {}

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
    OPTIONAL_MODEL_STATUS["XGBoost"] = "available"
except Exception as exc:
    XGBRegressor = None
    HAS_XGBOOST = False
    OPTIONAL_MODEL_STATUS["XGBoost"] = f"not available: {exc}"

try:
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
    OPTIONAL_MODEL_STATUS["LightGBM"] = "available"
except Exception as exc:
    LGBMRegressor = None
    HAS_LIGHTGBM = False
    OPTIONAL_MODEL_STATUS["LightGBM"] = f"not available: {exc}"

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
    OPTIONAL_MODEL_STATUS["CatBoost"] = "available"
except Exception as exc:
    CatBoostRegressor = None
    HAS_CATBOOST = False
    OPTIONAL_MODEL_STATUS["CatBoost"] = f"not available: {exc}"


# =============================================================================
# Configuration
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_EXCEL = PROJECT_DIR / "data.xlsx"
OUTPUT_DIR = PROJECT_DIR / "HydroPIML_RS_code_availability_outputs"

EXPERIMENT_TAG = "HydroPIML_RS_prediction"
PROPOSED_NAME = "HydroPIML-RS"

SHEET_NAME = "Daily_Data"

DATE_COL = "Date"
P_COL = "Precipitation_mm"
ET_COL = "Evapotranspiration_mm"
H_COL = "Water_level_m"
Q_COL = "Runoff_mm"

PRIMARY_HORIZONS = [1, 3]
HORIZONS = PRIMARY_HORIZONS

TRAIN_RATIO = 0.45
VAL_RATIO = 0.20
CAL_RATIO = 0.15

RANDOM_STATE = 42

LAGS = [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 30, 40]
ROLL_WINDOWS = [2, 3, 5, 7, 10, 14, 21, 30, 40]
API_WINDOWS = [3, 7, 14, 30]
API_DECAYS = [0.80, 0.85, 0.90, 0.95]

USE_XGBOOST_INTERNAL = True
USE_LIGHTGBM_INTERNAL = True
USE_CATBOOST_INTERNAL = True

ADD_INERTIA_SHRINKAGE_CANDIDATES = True
SHRINKAGE_LAMBDA_GRID = np.round(np.linspace(0.03, 1.00, 33), 3)
MIN_SHRINKAGE_LAMBDA = 0.03

ADD_DRY_LOW_SHRINKAGE_CANDIDATES = True
DRY_LOW_SHRINKAGE_TARGET_TYPES = ["Q"]
DRY_LOW_SHRINKAGE_LAMBDA_GRID = np.round(np.linspace(0.03, 1.00, 33), 3)
DRY_LOW_PSUM7_QUANTILE = 0.45
DRY_LOW_STATE_QUANTILE_Q = 0.60
DRY_LOW_STATE_QUANTILE_H = 0.60

ADD_METRIC_ORIENTED_STACKS = True
ADD_ADAPTIVE_LEADER_BLENDS = True
STACK_MAX_NONZERO = 4
STACK_L2 = 1.0e-4

ALLOW_Q_AFFINE_CALIBRATION = False
ALLOW_H_AFFINE_CALIBRATION = True
ALLOW_ISOTONIC_CALIBRATION = True
ALLOW_QUANTILE_MAPPING_CALIBRATION = True

CALIBRATION_SELECTION_USE_VAL_CAL = True
CALIBRATION_SELECTION_VAL_WEIGHT = 0.70
CALIBRATION_SELECTION_CAL_WEIGHT = 0.30

SAFE_CALIBRATION_ENABLE_IDENTITY_FALLBACK = True
SAFE_CALIBRATION_MIN_VAL_IMPROVEMENT_H = 0.015
SAFE_CALIBRATION_MIN_VAL_IMPROVEMENT_Q = 0.003

APPLY_QH_DIRECTION_CONSTRAINT = True
ADAPTIVE_QH_DIRECTION_CONSTRAINT = True
QH_CONSTRAINT_MAX_RELATIVE_DEGRADATION = 0.010

CONFORMAL_ALPHA = 0.10

WRITE_INTERNAL_SELECTION_AUDIT = True
WRITE_REFERENCE_SPLITS = True

EPS = 1.0e-12


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_sheet_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)
    return name[:31]


def json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=True, default=str)
    except Exception:
        return str(obj)


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_section(title: str) -> None:
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def safe_expm1(x: np.ndarray, clip_min: float = -50.0, clip_max: float = 50.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.expm1(np.clip(x, clip_min, clip_max))


def predict_1d(model: Any, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float).reshape(-1)


def physical_projection(pred: np.ndarray, target_name: str) -> np.ndarray:
    pred = np.asarray(pred, dtype=float).reshape(-1).copy()
    if target_name.upper().startswith("Q"):
        pred = np.maximum(pred, 0.0)
    return pred


def target_list_for_horizon(horizon: int) -> List[str]:
    if horizon == 1:
        return ["Q_point", "H_point"]
    return ["Q_point", "H_point", "Q_max", "H_max"]


def target_columns(target_name: str) -> Tuple[str, str]:
    if target_name == "Q_point":
        return "y_Q_point", "issue_state_Q"
    if target_name == "H_point":
        return "y_H_point", "issue_state_H"
    if target_name == "Q_max":
        return "y_Q_max", "issue_state_Q"
    if target_name == "H_max":
        return "y_H_max", "issue_state_H"
    raise ValueError(f"Unknown target_name: {target_name}")


def get_array_from_ref(ref: pd.DataFrame, col: str, default: Optional[float] = None) -> np.ndarray:
    if col in ref.columns:
        return pd.to_numeric(ref[col], errors="coerce").values.astype(float)
    if default is not None:
        return np.full(len(ref), default, dtype=float)
    return np.full(len(ref), np.nan, dtype=float)


# =============================================================================
# Data loading
# =============================================================================

def load_daily_data(input_excel: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if not input_excel.exists():
        raise FileNotFoundError(f"Input file not found: {input_excel}")

    excel = pd.ExcelFile(input_excel)
    sheet = SHEET_NAME if SHEET_NAME in excel.sheet_names else excel.sheet_names[0]

    df = pd.read_excel(input_excel, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]

    required = [DATE_COL, P_COL, ET_COL, H_COL, Q_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "The input file does not contain the required columns. "
            f"Missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[DATE_COL], errors="coerce"),
            "P": pd.to_numeric(df[P_COL], errors="coerce"),
            "ET": pd.to_numeric(df[ET_COL], errors="coerce"),
            "H": pd.to_numeric(df[H_COL], errors="coerce"),
            "Q": pd.to_numeric(df[Q_COL], errors="coerce"),
        }
    )

    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if out["Date"].duplicated().any():
        out = (
            out.groupby("Date", as_index=False)
            .agg({"P": "sum", "ET": "mean", "H": "mean", "Q": "mean"})
            .sort_values("Date")
            .reset_index(drop=True)
        )

    for col in ["P", "ET", "H", "Q"]:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        if out[col].isna().any():
            out[col] = out[col].interpolate(limit_direction="both")
        if out[col].isna().any():
            raise ValueError(f"Column {col} still contains missing values after interpolation.")

    selected = {
        "Date": DATE_COL,
        "P": P_COL,
        "ET": ET_COL,
        "H": H_COL,
        "Q": Q_COL,
    }

    return out, selected


# =============================================================================
# Feature engineering
# =============================================================================

def rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=w).sum().values


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=w).mean().values


def rolling_max(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=w).max().values


def rolling_min(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=w).min().values


def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=w).std().values


def rolling_wetdays(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(np.asarray(x) > 0.1).rolling(w, min_periods=w).sum().values


def antecedent_precip_index(P: np.ndarray, window: int = 7, decay: float = 0.9) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    out = np.full_like(P, np.nan, dtype=float)

    for t in range(len(P)):
        if t - window + 1 < 0:
            continue
        out[t] = sum((decay ** i) * P[t - i] for i in range(window))

    return out


def construct_feature_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    F = pd.DataFrame(index=df.index)

    date = pd.to_datetime(df["Date"])
    doy = date.dt.dayofyear.astype(float)
    month = date.dt.month.astype(float)

    F["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    F["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    F["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    F["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    F["time_index"] = np.arange(len(df), dtype=float) / max(len(df) - 1, 1)

    P = df["P"].astype(float).values
    ET = df["ET"].astype(float).values
    H = df["H"].astype(float).values
    Q = df["Q"].astype(float).values

    P_minus_ET = P - ET
    effective_P = np.maximum(P - ET, 0.0)

    raw_series = {
        "P": P,
        "ET": ET,
        "H": H,
        "Q": Q,
        "P_minus_ET": P_minus_ET,
        "effective_P": effective_P,
        "log1p_Q": np.log1p(np.maximum(Q, 0.0)),
        "log1p_P": np.log1p(np.maximum(P, 0.0)),
    }

    for name, arr in raw_series.items():
        s = pd.Series(arr)
        for lag in LAGS:
            F[f"{name}_lag{lag}"] = s.shift(lag).values

    for w in ROLL_WINDOWS:
        F[f"P_sum_{w}"] = rolling_sum(P, w)
        F[f"P_mean_{w}"] = rolling_mean(P, w)
        F[f"P_max_{w}"] = rolling_max(P, w)
        F[f"P_std_{w}"] = rolling_std(P, w)
        F[f"P_wetdays_{w}"] = rolling_wetdays(P, w)
        F[f"P_concentration_{w}"] = F[f"P_max_{w}"] / (F[f"P_sum_{w}"] + EPS)

        F[f"ET_sum_{w}"] = rolling_sum(ET, w)
        F[f"ET_mean_{w}"] = rolling_mean(ET, w)
        F[f"ET_max_{w}"] = rolling_max(ET, w)

        F[f"P_minus_ET_sum_{w}"] = rolling_sum(P_minus_ET, w)
        F[f"P_minus_ET_mean_{w}"] = rolling_mean(P_minus_ET, w)

        F[f"effective_P_sum_{w}"] = rolling_sum(effective_P, w)
        F[f"effective_P_mean_{w}"] = rolling_mean(effective_P, w)

        F[f"H_mean_{w}"] = rolling_mean(H, w)
        F[f"H_max_{w}"] = rolling_max(H, w)
        F[f"H_min_{w}"] = rolling_min(H, w)
        F[f"H_std_{w}"] = rolling_std(H, w)

        F[f"Q_sum_{w}"] = rolling_sum(Q, w)
        F[f"Q_mean_{w}"] = rolling_mean(Q, w)
        F[f"Q_max_{w}"] = rolling_max(Q, w)
        F[f"Q_min_{w}"] = rolling_min(Q, w)
        F[f"Q_std_{w}"] = rolling_std(Q, w)

        F[f"log1p_Q_mean_{w}"] = rolling_mean(np.log1p(np.maximum(Q, 0.0)), w)
        F[f"log1p_Q_max_{w}"] = rolling_max(np.log1p(np.maximum(Q, 0.0)), w)

        F[f"runoff_coeff_sum_{w}"] = F[f"Q_sum_{w}"] / (F[f"P_sum_{w}"] + EPS)
        F[f"QH_response_ratio_{w}"] = F[f"Q_mean_{w}"] / (np.abs(F[f"H_mean_{w}"]) + EPS)

    for w in API_WINDOWS:
        for decay in API_DECAYS:
            ds = str(decay).replace(".", "p")
            F[f"API_P_w{w}_d{ds}"] = antecedent_precip_index(P, w, decay)

    for lag in [1, 2, 3, 5, 7, 14]:
        F[f"dQ_lag{lag}"] = pd.Series(Q).diff(lag).values
        F[f"dH_lag{lag}"] = pd.Series(H).diff(lag).values
        F[f"dP_lag{lag}"] = pd.Series(P).diff(lag).values
        F[f"dET_lag{lag}"] = pd.Series(ET).diff(lag).values
        F[f"dP_minus_ET_lag{lag}"] = pd.Series(P_minus_ET).diff(lag).values

        F[f"rel_dQ_lag{lag}"] = F[f"dQ_lag{lag}"] / (
            np.abs(pd.Series(Q).shift(lag).values) + EPS
        )
        F[f"rel_dH_lag{lag}"] = F[f"dH_lag{lag}"] / (
            np.abs(pd.Series(H).shift(lag).values) + EPS
        )

    F["P_lag0_x_H_lag0"] = F["P_lag0"] * F["H_lag0"]
    F["P_lag0_x_Q_lag0"] = F["P_lag0"] * F["Q_lag0"]
    F["P_minus_ET_lag0_x_Q_lag0"] = F["P_minus_ET_lag0"] * F["Q_lag0"]
    F["P_minus_ET_lag0_x_H_lag0"] = F["P_minus_ET_lag0"] * F["H_lag0"]

    if "API_P_w7_d0p9" in F.columns:
        F["API7_090_x_H"] = F["API_P_w7_d0p9"] * F["H_lag0"]
        F["API7_090_x_Q"] = F["API_P_w7_d0p9"] * F["Q_lag0"]

    if "API_P_w14_d0p9" in F.columns:
        F["API14_090_x_H"] = F["API_P_w14_d0p9"] * F["H_lag0"]
        F["API14_090_x_Q"] = F["API_P_w14_d0p9"] * F["Q_lag0"]

    for w in [3, 7, 14, 30]:
        F[f"is_dry_{w}"] = (F[f"P_sum_{w}"] <= 0.1).astype(float)
        F[f"is_wet_{w}"] = (F[f"P_sum_{w}"] > 0.1).astype(float)
        F[f"P7_low_proxy_{w}"] = F[f"P_sum_{w}"] / (
            F[f"P_sum_{w}"].rolling(60, min_periods=10).mean() + EPS
        )

    F = F.replace([np.inf, -np.inf], np.nan)

    return F, list(F.columns)


# =============================================================================
# Supervised dataset and chronological split
# =============================================================================

def future_window_max(arr: np.ndarray, horizon: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    cols = []

    for k in range(1, horizon + 1):
        cols.append(pd.Series(arr).shift(-k).values)

    M = np.column_stack(cols)
    valid = np.all(np.isfinite(M), axis=1)

    out = np.full(len(arr), np.nan, dtype=float)
    out[valid] = np.max(M[valid], axis=1)

    return out


def make_supervised_dataset(
    df: pd.DataFrame,
    feature_table: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
) -> pd.DataFrame:
    X_all = feature_table[feature_cols].copy()

    base = pd.DataFrame(
        {
            "row_id": np.arange(len(df)),
            "issue_date": df["Date"].values,
            "pred_date": df["Date"].shift(-horizon).values,
            "y_Q_point": df["Q"].shift(-horizon).values,
            "y_H_point": df["H"].shift(-horizon).values,
            "y_Q_max": future_window_max(df["Q"].values, horizon),
            "y_H_max": future_window_max(df["H"].values, horizon),
            "issue_state_Q": df["Q"].values,
            "issue_state_H": df["H"].values,
            "P_issue": df["P"].values,
            "ET_issue": df["ET"].values,
            "Q_issue": df["Q"].values,
            "H_issue": df["H"].values,
        }
    )

    full = pd.concat([base, X_all], axis=1)

    required = feature_cols + [
        "pred_date",
        "y_Q_point",
        "y_H_point",
        "y_Q_max",
        "y_H_max",
        "issue_state_Q",
        "issue_state_H",
    ]

    full = full.dropna(subset=required).reset_index(drop=True)

    return full


def time_ordered_split4(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    n_cal = int(round(n * CAL_RATIO))
    n_test = n - n_train - n_val - n_cal

    if min(n_train, n_val, n_cal, n_test) <= 10:
        raise ValueError(
            f"Invalid split: n={n}, train={n_train}, "
            f"validation={n_val}, calibration={n_cal}, test={n_test}"
        )

    idx_train = np.arange(0, n_train)
    idx_val = np.arange(n_train, n_train + n_val)
    idx_cal = np.arange(n_train + n_val, n_train + n_val + n_cal)
    idx_test = np.arange(n_train + n_val + n_cal, n)

    return idx_train, idx_val, idx_cal, idx_test


def build_reference_split(supervised: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    g = supervised.iloc[idx].copy().reset_index(drop=True)

    ref = pd.DataFrame(
        {
            "issue_date": g["issue_date"],
            "pred_date": g["pred_date"],
            "Observed_Q_point": g["y_Q_point"],
            "Observed_H_point": g["y_H_point"],
            "Observed_Q_max": g["y_Q_max"],
            "Observed_H_max": g["y_H_max"],
            "issue_state_Q": g["issue_state_Q"],
            "issue_state_H": g["issue_state_H"],
            "P_issue": g["P_issue"],
            "ET_issue": g["ET_issue"],
            "Q_issue": g["Q_issue"],
            "H_issue": g["H_issue"],
        }
    )

    for w in [2, 3, 5, 7, 14, 21, 30, 40]:
        for prefix in ["P_sum", "ET_sum", "P_minus_ET_sum", "effective_P_sum"]:
            c = f"{prefix}_{w}"
            if c in g.columns:
                ref[f"{c}_issue"] = g[c]

    return ref


def split_distribution_diagnostics(horizon: int, split_name: str, ref: pd.DataFrame) -> pd.DataFrame:
    row: Dict[str, Any] = {
        "horizon": horizon,
        "split": split_name,
        "N": len(ref),
        "pred_date_start": ref["pred_date"].min(),
        "pred_date_end": ref["pred_date"].max(),
    }

    cols = [
        "P_issue",
        "ET_issue",
        "Q_issue",
        "H_issue",
        "Observed_Q_point",
        "Observed_H_point",
        "Observed_Q_max",
        "Observed_H_max",
    ]

    for c in cols:
        if c not in ref.columns:
            continue

        x = pd.to_numeric(ref[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
        if len(x) == 0:
            continue

        row[f"{c}_mean"] = float(np.mean(x))
        row[f"{c}_std"] = float(np.std(x))
        row[f"{c}_p50"] = float(np.quantile(x, 0.50))
        row[f"{c}_p90"] = float(np.quantile(x, 0.90))
        row[f"{c}_p95"] = float(np.quantile(x, 0.95))
        row[f"{c}_max"] = float(np.max(x))

        if c == "P_issue":
            row["P_issue_sum"] = float(np.sum(x))
            row["wet_day_rate_issue"] = float(np.mean(x > 0.1))
            row["dry_day_rate_issue"] = float(np.mean(x <= 0.1))

    for c in [
        "P_sum_3_issue",
        "P_sum_7_issue",
        "P_sum_14_issue",
        "P_sum_30_issue",
        "effective_P_sum_7_issue",
        "P_minus_ET_sum_7_issue",
    ]:
        if c not in ref.columns:
            continue

        x = pd.to_numeric(ref[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
        if len(x) == 0:
            continue

        row[f"{c}_mean"] = float(np.mean(x))
        row[f"{c}_p50"] = float(np.quantile(x, 0.50))
        row[f"{c}_p90"] = float(np.quantile(x, 0.90))
        row[f"{c}_p95"] = float(np.quantile(x, 0.95))
        row[f"{c}_max"] = float(np.max(x))

    return pd.DataFrame([row])


# =============================================================================
# Metrics
# =============================================================================

def hydro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)

    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]

    if len(y) == 0:
        return {
            "N": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "NSE": np.nan,
            "KGE": np.nan,
            "Bias": np.nan,
            "PBIAS_percent": np.nan,
            "PeakAbsError": np.nan,
        }

    err = p - y

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    pbias = float(100.0 * np.sum(err) / (np.sum(y) + EPS))

    den = np.sum((y - np.mean(y)) ** 2)
    nse = float(1.0 - np.sum(err ** 2) / (den + EPS))

    if len(y) >= 3 and np.std(y) > EPS and np.std(p) > EPS and abs(np.mean(y)) > EPS:
        r = float(np.corrcoef(y, p)[0, 1])
        alpha = float(np.std(p) / (np.std(y) + EPS))
        beta = float(np.mean(p) / (np.mean(y) + EPS))
        kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    else:
        kge = np.nan

    peak_idx = int(np.argmax(y))
    peak_abs_error = float(abs(p[peak_idx] - y[peak_idx]))

    return {
        "N": int(len(y)),
        "MAE": mae,
        "RMSE": rmse,
        "NSE": nse,
        "KGE": kge,
        "Bias": bias,
        "PBIAS_percent": pbias,
        "PeakAbsError": peak_abs_error,
    }


def low_flow_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: Optional[float] = None,
    quantile: float = 0.50,
) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]

    if len(y) == 0:
        return np.nan

    if threshold is None:
        threshold = np.nanquantile(y, quantile)

    mask = y <= threshold
    if np.sum(mask) == 0:
        return np.nan

    return float(np.mean(np.abs(p[mask] - y[mask])))


def high_flow_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: Optional[float] = None,
    quantile: float = 0.90,
) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]

    if len(y) == 0:
        return np.nan

    if threshold is None:
        threshold = np.nanquantile(y, quantile)

    mask = y >= threshold
    if np.sum(mask) == 0:
        return np.nan

    return float(np.mean(np.abs(p[mask] - y[mask])))


def masked_metric(y: np.ndarray, p: np.ndarray, mask: np.ndarray, metric_name: str) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if np.sum(mask) < 5:
        return np.nan

    met = hydro_metrics(y[mask], p[mask])
    return met.get(metric_name, np.nan)


def masked_mae(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    m = mask & np.isfinite(y) & np.isfinite(p)

    if np.sum(m) < 5:
        return np.nan

    return float(np.mean(np.abs(p[m] - y[m])))


def masked_positive_bias_ratio(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    m = mask & np.isfinite(y) & np.isfinite(p)

    if np.sum(m) < 5:
        return np.nan

    pos_bias = np.maximum(p[m] - y[m], 0.0)
    return float(np.mean(pos_bias) / (np.mean(np.abs(y[m])) + EPS))


def selection_objective_value(
    y: np.ndarray,
    p: np.ndarray,
    target_name: str,
    profile: str = "balanced",
) -> float:
    met = hydro_metrics(y, p)

    rmse = met["RMSE"] if np.isfinite(met["RMSE"]) else 1.0e9
    mae = met["MAE"] if np.isfinite(met["MAE"]) else 1.0e9
    nse = met["NSE"] if np.isfinite(met["NSE"]) else -1.0e9
    kge = met["KGE"] if np.isfinite(met["KGE"]) else -1.0e9
    pbias = abs(met["PBIAS_percent"]) if np.isfinite(met["PBIAS_percent"]) else 1.0e9
    peak = met["PeakAbsError"] if np.isfinite(met["PeakAbsError"]) else 1.0e9

    y = np.asarray(y, dtype=float)
    scale = np.nanstd(y) + EPS

    if profile == "rmse_nse":
        return 1.00 * rmse + 0.30 * mae + 0.80 * max(0.0, 1.0 - nse) * scale + 0.02 * pbias * scale

    if profile == "mae":
        return 1.00 * mae + 0.35 * rmse + 0.05 * pbias * scale

    if profile == "kge_bias":
        return 0.50 * rmse + 0.40 * mae + 0.90 * max(0.0, 1.0 - kge) * scale + 0.12 * pbias * scale

    if profile == "peak":
        return 0.50 * rmse + 0.30 * mae + 0.25 * peak + 0.35 * max(0.0, 1.0 - nse) * scale

    if profile == "dry_low_mae":
        return 1.10 * mae + 0.35 * rmse + 0.10 * pbias * scale

    return (
        0.70 * rmse
        + 0.50 * mae
        + 0.40 * max(0.0, 1.0 - nse) * scale
        + 0.35 * max(0.0, 1.0 - kge) * scale
        + 0.05 * pbias * scale
    )


# =============================================================================
# Internal learner components
# =============================================================================

def build_internal_learners(random_state: int = 42) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    models["Ridge"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=random_state)),
        ]
    )

    models["ElasticNet"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=0.001,
                    l1_ratio=0.30,
                    max_iter=20000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    models["Huber"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=2000)),
        ]
    )

    models["PLSRegression"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", PLSRegression(n_components=10)),
        ]
    )

    models["SVR_RBF"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale")),
        ]
    )

    models["RandomForest"] = RandomForestRegressor(
        n_estimators=800,
        max_features="sqrt",
        min_samples_leaf=2,
        max_depth=None,
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1,
    )

    models["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=900,
        max_features="sqrt",
        min_samples_leaf=2,
        max_depth=None,
        bootstrap=False,
        random_state=random_state,
        n_jobs=-1,
    )

    models["GradientBoosting"] = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.025,
        max_depth=3,
        subsample=0.80,
        random_state=random_state,
    )

    models["HistGradientBoosting"] = HistGradientBoostingRegressor(
        max_iter=450,
        learning_rate=0.025,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        early_stopping=True,
        random_state=random_state,
    )

    if USE_XGBOOST_INTERNAL and HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=900,
            max_depth=3,
            learning_rate=0.025,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )

    if USE_LIGHTGBM_INTERNAL and HAS_LIGHTGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.020,
            num_leaves=15,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_lambda=1.0,
            min_child_samples=5,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    if USE_CATBOOST_INTERNAL and HAS_CATBOOST:
        models["CatBoost"] = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.020,
            depth=4,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )

    return models


def fit_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Any:
    m = clone(model)

    if sample_weight is None:
        m.fit(X, y)
        return m

    if isinstance(m, Pipeline):
        final_step = m.steps[-1][0]
        try:
            m.fit(X, y, **{f"{final_step}__sample_weight": sample_weight})
            return m
        except Exception:
            pass

    try:
        m.fit(X, y, sample_weight=sample_weight)
    except Exception:
        m.fit(X, y)

    return m


def sample_weight_for_target(y: np.ndarray, target_name: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y, dtype=float)

    q70 = np.nanquantile(y, 0.70)
    q85 = np.nanquantile(y, 0.85)
    q95 = np.nanquantile(y, 0.95)

    if target_name.upper().startswith("Q"):
        w += 0.80 * (y >= q70)
        w += 1.50 * (y >= q85)
        w += 2.00 * (y >= q95)
        w += np.maximum(y, 0.0) / (np.nanquantile(y, 0.90) + EPS)
        return np.clip(w, 1.0, 7.0)

    w += 0.50 * (y >= q70)
    w += 1.00 * (y >= q85)
    w += 1.50 * (y >= q95)
    return np.clip(w, 1.0, 5.0)


# =============================================================================
# Calibration
# =============================================================================

def make_quantile_mapping(y: np.ndarray, p: np.ndarray, target_name: str):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]

    if len(y) < 15:
        return None

    qs = np.linspace(0.02, 0.98, 49)
    p_q = np.quantile(p, qs)
    y_q = np.quantile(y, qs)

    p_q_unique, idx = np.unique(p_q, return_index=True)
    y_q_unique = y_q[idx]

    if len(p_q_unique) < 5:
        return None

    def apply_qm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        out = np.interp(x, p_q_unique, y_q_unique, left=y_q_unique[0], right=y_q_unique[-1])
        return physical_projection(out, target_name)

    return apply_qm, {
        "quantiles": len(p_q_unique),
        "q_min": float(qs[0]),
        "q_max": float(qs[-1]),
    }


def fit_auto_calibration(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    target_name: str,
    y_val: Optional[np.ndarray] = None,
    p_val: Optional[np.ndarray] = None,
    horizon: Optional[int] = None,
):
    y_cal = np.asarray(y_cal, dtype=float).reshape(-1)
    p_cal = np.asarray(p_cal, dtype=float).reshape(-1)

    m = np.isfinite(y_cal) & np.isfinite(p_cal)
    y = y_cal[m]
    p = p_cal[m]

    methods: Dict[str, Dict[str, Any]] = {}

    def apply_identity(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float).reshape(-1)

    methods["identity"] = {"apply": apply_identity, "params": {}}

    if len(y) >= 5:
        bias = float(np.mean(y - p))

        def apply_bias(x: np.ndarray, bias: float = bias) -> np.ndarray:
            return np.asarray(x, dtype=float).reshape(-1) + bias

        methods["bias_correction"] = {
            "apply": apply_bias,
            "params": {"bias": bias},
        }

    allow_affine = (
        (target_name.upper().startswith("Q") and ALLOW_Q_AFFINE_CALIBRATION)
        or ((not target_name.upper().startswith("Q")) and ALLOW_H_AFFINE_CALIBRATION)
    )

    if allow_affine and len(y) >= 5 and np.std(p) > EPS:
        lr = LinearRegression()
        lr.fit(p.reshape(-1, 1), y)
        a = float(lr.coef_[0])
        b = float(lr.intercept_)

        if not (target_name.upper().startswith("Q") and a < 0.0):
            def apply_affine(x: np.ndarray, a: float = a, b: float = b) -> np.ndarray:
                return a * np.asarray(x, dtype=float).reshape(-1) + b

            methods["affine"] = {
                "apply": apply_affine,
                "params": {"a": a, "b": b},
            }

    if len(y) >= 5 and np.sum(p ** 2) > EPS:
        slope = float(np.sum(p * y) / (np.sum(p ** 2) + EPS))

        if not (target_name.upper().startswith("Q") and slope < 0.0):
            def apply_slope(x: np.ndarray, slope: float = slope) -> np.ndarray:
                return slope * np.asarray(x, dtype=float).reshape(-1)

            methods["slope_only"] = {
                "apply": apply_slope,
                "params": {"slope": slope},
            }

    if target_name.upper().startswith("Q") and len(y) >= 5:
        y_pos = np.maximum(y, 0.0)
        p_pos = np.maximum(p, 0.0)

        log_bias = float(np.mean(np.log1p(y_pos) - np.log1p(p_pos)))

        def apply_log_bias(x: np.ndarray, log_bias: float = log_bias) -> np.ndarray:
            x = np.maximum(np.asarray(x, dtype=float).reshape(-1), 0.0)
            return safe_expm1(np.log1p(x) + log_bias)

        methods["log1p_bias_correction"] = {
            "apply": apply_log_bias,
            "params": {"log_bias": log_bias},
        }

        if np.sum(p_pos) > EPS:
            ratio_sum = float(np.sum(y_pos) / (np.sum(p_pos) + EPS))

            def apply_sum_ratio(x: np.ndarray, ratio_sum: float = ratio_sum) -> np.ndarray:
                return ratio_sum * np.asarray(x, dtype=float).reshape(-1)

            methods["sum_ratio"] = {
                "apply": apply_sum_ratio,
                "params": {"ratio_sum": ratio_sum},
            }

    if ALLOW_ISOTONIC_CALIBRATION and len(y) >= 20 and np.std(p) > EPS:
        try:
            iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
            iso.fit(p, y)

            def apply_iso(x: np.ndarray, iso_model: IsotonicRegression = iso) -> np.ndarray:
                x = np.asarray(x, dtype=float).reshape(-1)
                return iso_model.predict(x)

            methods["isotonic_monotone"] = {
                "apply": apply_iso,
                "params": {"out_of_bounds": "clip", "increasing": True},
            }
        except Exception:
            pass

    if ALLOW_QUANTILE_MAPPING_CALIBRATION and len(y) >= 20 and np.std(p) > EPS:
        qm = make_quantile_mapping(y, p, target_name)
        if qm is not None:
            apply_qm, qm_params = qm
            methods["quantile_mapping"] = {
                "apply": apply_qm,
                "params": qm_params,
            }

    has_val = (
        y_val is not None
        and p_val is not None
        and len(np.asarray(y_val).reshape(-1)) == len(np.asarray(p_val).reshape(-1))
        and CALIBRATION_SELECTION_USE_VAL_CAL
    )

    if has_val:
        y_val = np.asarray(y_val, dtype=float).reshape(-1)
        p_val = np.asarray(p_val, dtype=float).reshape(-1)

    rows: List[Dict[str, Any]] = []

    cal_low_thr = np.nanquantile(y_cal, 0.50)
    cal_high_thr = np.nanquantile(y_cal, 0.90)

    if has_val:
        val_low_thr = np.nanquantile(y_val, 0.50)
        val_high_thr = np.nanquantile(y_val, 0.90)

    for method_name, obj in methods.items():
        pred_cal = physical_projection(obj["apply"](p_cal), target_name)
        met_cal = hydro_metrics(y_cal, pred_cal)

        row: Dict[str, Any] = {
            "calibration_method": method_name,
            "calibration_params": json_dumps(obj["params"]),
            "cal_MAE": met_cal["MAE"],
            "cal_RMSE": met_cal["RMSE"],
            "cal_NSE": met_cal["NSE"],
            "cal_KGE": met_cal["KGE"],
            "cal_Bias": met_cal["Bias"],
            "cal_PBIAS_percent": met_cal["PBIAS_percent"],
            "cal_abs_PBIAS": abs(met_cal["PBIAS_percent"]),
            "cal_low_flow_MAE": low_flow_mae(y_cal, pred_cal, threshold=cal_low_thr),
            "cal_high_flow_MAE": high_flow_mae(y_cal, pred_cal, threshold=cal_high_thr),
            "MAE": met_cal["MAE"],
            "RMSE": met_cal["RMSE"],
            "NSE": met_cal["NSE"],
            "KGE": met_cal["KGE"],
            "Bias": met_cal["Bias"],
            "PBIAS_percent": met_cal["PBIAS_percent"],
            "PeakAbsError": met_cal["PeakAbsError"],
            "N": met_cal["N"],
        }

        if has_val:
            pred_val = physical_projection(obj["apply"](p_val), target_name)
            met_val = hydro_metrics(y_val, pred_val)

            row.update(
                {
                    "val_MAE": met_val["MAE"],
                    "val_RMSE": met_val["RMSE"],
                    "val_NSE": met_val["NSE"],
                    "val_KGE": met_val["KGE"],
                    "val_Bias": met_val["Bias"],
                    "val_PBIAS_percent": met_val["PBIAS_percent"],
                    "val_abs_PBIAS": abs(met_val["PBIAS_percent"]),
                    "val_low_flow_MAE": low_flow_mae(y_val, pred_val, threshold=val_low_thr),
                    "val_high_flow_MAE": high_flow_mae(y_val, pred_val, threshold=val_high_thr),
                }
            )

        rows.append(row)

    metrics_df = pd.DataFrame(rows)

    def add_rank_score(df: pd.DataFrame, prefix: str, weight_prefix: str) -> np.ndarray:
        score = np.zeros(len(df), dtype=float)

        if target_name == "Q_point":
            metric_weights = {
                "RMSE": 0.28,
                "MAE": 0.18,
                "NSE": 0.18,
                "KGE": 0.16,
                "abs_PBIAS": 0.10,
                "low_flow_MAE": 0.06,
                "high_flow_MAE": 0.04,
            }
        elif target_name == "Q_max":
            metric_weights = {
                "RMSE": 0.24,
                "MAE": 0.14,
                "NSE": 0.17,
                "KGE": 0.20,
                "abs_PBIAS": 0.08,
                "low_flow_MAE": 0.04,
                "high_flow_MAE": 0.13,
            }
        else:
            metric_weights = {
                "RMSE": 0.24,
                "MAE": 0.22,
                "NSE": 0.18,
                "KGE": 0.18,
                "abs_PBIAS": 0.08,
                "low_flow_MAE": 0.06,
                "high_flow_MAE": 0.04,
            }

        for metric, weight in metric_weights.items():
            col = f"{prefix}_{metric}"
            if col not in df.columns:
                continue

            ascending = True
            if metric in ["NSE", "KGE"]:
                ascending = False

            rank_col = f"rank_{weight_prefix}_{metric}"
            df[rank_col] = df[col].rank(method="average", ascending=ascending, na_option="bottom")
            score += weight * df[rank_col].values

        return score

    if has_val:
        metrics_df["cal_rank_score"] = add_rank_score(metrics_df, "cal", "cal")
        metrics_df["val_rank_score"] = add_rank_score(metrics_df, "val", "val")
        metrics_df["calibration_selection_score"] = (
            CALIBRATION_SELECTION_VAL_WEIGHT * metrics_df["val_rank_score"]
            + CALIBRATION_SELECTION_CAL_WEIGHT * metrics_df["cal_rank_score"]
        )
    else:
        metrics_df["calibration_selection_score"] = metrics_df["RMSE"].rank(method="average", ascending=True)

    selected = str(metrics_df.sort_values("calibration_selection_score").iloc[0]["calibration_method"])

    fallback_triggered = False
    fallback_reason = None

    if SAFE_CALIBRATION_ENABLE_IDENTITY_FALLBACK and has_val and "identity" in methods:
        selected_apply_tmp = methods[selected]["apply"]
        identity_apply_tmp = methods["identity"]["apply"]

        pred_val_selected = physical_projection(selected_apply_tmp(p_val), target_name)
        pred_val_identity = physical_projection(identity_apply_tmp(p_val), target_name)

        if target_name.upper().startswith("Q"):
            profile = "rmse_nse"
            min_improve = SAFE_CALIBRATION_MIN_VAL_IMPROVEMENT_Q
        else:
            profile = "balanced"
            min_improve = SAFE_CALIBRATION_MIN_VAL_IMPROVEMENT_H

        obj_selected = selection_objective_value(y_val, pred_val_selected, target_name, profile=profile)
        obj_identity = selection_objective_value(y_val, pred_val_identity, target_name, profile=profile)

        required_obj = obj_identity * (1.0 - min_improve)

        if selected != "identity" and obj_selected > required_obj:
            selected = "identity"
            fallback_triggered = True
            fallback_reason = (
                f"non_identity_calibration_did_not_improve_validation_enough; "
                f"obj_selected={obj_selected:.8g}; "
                f"obj_identity={obj_identity:.8g}; "
                f"required_obj<={required_obj:.8g}"
            )

    selected_apply_raw = methods[selected]["apply"]

    def selected_apply(x: np.ndarray) -> np.ndarray:
        return physical_projection(selected_apply_raw(x), target_name)

    info = {
        "selected_calibration_method": selected,
        "selected_calibration_params": methods[selected]["params"],
        "selection_policy": "safe_validation_plus_calibration_calibration_selection",
        "calibration_fitted_on": "calibration_period_only",
        "calibration_selected_on": "validation_plus_calibration" if has_val else "calibration_only",
        "calibration_selection_val_weight": CALIBRATION_SELECTION_VAL_WEIGHT if has_val else None,
        "calibration_selection_cal_weight": CALIBRATION_SELECTION_CAL_WEIGHT if has_val else None,
        "safe_identity_fallback_triggered": fallback_triggered,
        "safe_identity_fallback_reason": fallback_reason,
        "allow_Q_affine_calibration": ALLOW_Q_AFFINE_CALIBRATION,
        "allow_H_affine_calibration": ALLOW_H_AFFINE_CALIBRATION,
        "allow_isotonic_calibration": ALLOW_ISOTONIC_CALIBRATION,
        "allow_quantile_mapping_calibration": ALLOW_QUANTILE_MAPPING_CALIBRATION,
    }

    metrics_df = metrics_df.sort_values("calibration_selection_score").reset_index(drop=True)

    return selected_apply, info, metrics_df


# =============================================================================
# Candidate generation
# =============================================================================

def apply_inertia_shrinkage(
    base_pred: np.ndarray,
    issue_state: np.ndarray,
    target_name: str,
    lam: float,
) -> np.ndarray:
    base_pred = np.asarray(base_pred, dtype=float).reshape(-1)
    issue_state = np.asarray(issue_state, dtype=float).reshape(-1)
    lam = float(lam)

    if target_name.upper().startswith("Q"):
        base_pos = np.maximum(base_pred, 0.0)
        state_pos = np.maximum(issue_state, 0.0)
        out = safe_expm1(
            np.log1p(state_pos)
            + lam * (np.log1p(base_pos) - np.log1p(state_pos))
        )
        return physical_projection(out, target_name)

    out = issue_state + lam * (base_pred - issue_state)
    return physical_projection(out, target_name)


def optimize_shrinkage_lambda(
    y_val: np.ndarray,
    base_val: np.ndarray,
    state_val: np.ndarray,
    target_name: str,
) -> Tuple[float, float]:
    best_lam = 1.0
    best_obj = np.inf

    for lam in SHRINKAGE_LAMBDA_GRID:
        if lam < MIN_SHRINKAGE_LAMBDA:
            continue

        pred = apply_inertia_shrinkage(base_val, state_val, target_name, lam)
        obj = selection_objective_value(y_val, pred, target_name, profile="balanced")

        if obj < best_obj:
            best_obj = obj
            best_lam = float(lam)

    return best_lam, float(best_obj)


def dry_low_masks_from_refs(
    ref_train: pd.DataFrame,
    ref_val: pd.DataFrame,
    ref_cal: pd.DataFrame,
    ref_test: pd.DataFrame,
    state_train: np.ndarray,
    state_val: np.ndarray,
    state_cal: np.ndarray,
    state_test: np.ndarray,
    target_name: str,
):
    p7_val = get_array_from_ref(ref_val, "P_sum_7_issue")
    p7_cal = get_array_from_ref(ref_cal, "P_sum_7_issue")

    if np.sum(np.isfinite(p7_val)) < 10 or np.sum(np.isfinite(p7_cal)) < 10:
        return None

    p7_ref = np.concatenate([p7_val[np.isfinite(p7_val)], p7_cal[np.isfinite(p7_cal)]])
    p7_thr = float(np.nanquantile(p7_ref, DRY_LOW_PSUM7_QUANTILE))

    state_ref = np.concatenate(
        [
            np.asarray(state_val, dtype=float),
            np.asarray(state_cal, dtype=float),
        ]
    )

    if target_name.upper().startswith("Q"):
        state_thr = float(np.nanquantile(state_ref, DRY_LOW_STATE_QUANTILE_Q))
    else:
        state_thr = float(np.nanquantile(state_ref, DRY_LOW_STATE_QUANTILE_H))

    def make_mask(ref: pd.DataFrame, state: np.ndarray) -> np.ndarray:
        p7 = get_array_from_ref(ref, "P_sum_7_issue")
        state = np.asarray(state, dtype=float)
        return np.isfinite(p7) & np.isfinite(state) & (p7 <= p7_thr) & (state <= state_thr)

    return {
        "train": make_mask(ref_train, state_train),
        "val": make_mask(ref_val, state_val),
        "cal": make_mask(ref_cal, state_cal),
        "test": make_mask(ref_test, state_test),
        "p7_threshold": p7_thr,
        "state_threshold": state_thr,
    }


def apply_conditional_dry_low_shrinkage(
    base_pred: np.ndarray,
    issue_state: np.ndarray,
    target_name: str,
    dry_low_mask: np.ndarray,
    lam_dry: float,
) -> np.ndarray:
    base_pred = np.asarray(base_pred, dtype=float).reshape(-1)
    issue_state = np.asarray(issue_state, dtype=float).reshape(-1)
    dry_low_mask = np.asarray(dry_low_mask, dtype=bool)

    lam = np.ones_like(base_pred, dtype=float)
    lam[dry_low_mask] = float(lam_dry)

    if target_name.upper().startswith("Q"):
        base_pos = np.maximum(base_pred, 0.0)
        state_pos = np.maximum(issue_state, 0.0)
        out = safe_expm1(
            np.log1p(state_pos)
            + lam * (np.log1p(base_pos) - np.log1p(state_pos))
        )
        return physical_projection(out, target_name)

    out = issue_state + lam * (base_pred - issue_state)
    return physical_projection(out, target_name)


def optimize_dry_low_shrinkage_lambda(
    y_val: np.ndarray,
    y_cal: np.ndarray,
    base_val: np.ndarray,
    base_cal: np.ndarray,
    state_val: np.ndarray,
    state_cal: np.ndarray,
    target_name: str,
    mask_val: np.ndarray,
    mask_cal: np.ndarray,
    horizon: int,
):
    y_opt = np.concatenate([np.asarray(y_val), np.asarray(y_cal)])
    base_opt = np.concatenate([np.asarray(base_val), np.asarray(base_cal)])
    state_opt = np.concatenate([np.asarray(state_val), np.asarray(state_cal)])
    mask_opt = np.concatenate([np.asarray(mask_val, dtype=bool), np.asarray(mask_cal, dtype=bool)])

    if np.sum(mask_opt) < 8:
        return None, None

    best_lam = 1.0
    best_obj = np.inf

    for lam in DRY_LOW_SHRINKAGE_LAMBDA_GRID:
        pred = apply_conditional_dry_low_shrinkage(
            base_pred=base_opt,
            issue_state=state_opt,
            target_name=target_name,
            dry_low_mask=mask_opt,
            lam_dry=lam,
        )

        obj_all = selection_objective_value(y_opt, pred, target_name, profile="balanced")
        obj_low = selection_objective_value(y_opt[mask_opt], pred[mask_opt], target_name, profile="dry_low_mae")

        if target_name == "Q_point" and horizon == 1:
            obj = 0.55 * obj_all + 0.45 * obj_low
        else:
            obj = 0.70 * obj_all + 0.30 * obj_low

        if obj < best_obj:
            best_obj = obj
            best_lam = float(lam)

    return best_lam, float(best_obj)


def fit_simplex_weights(
    P: np.ndarray,
    y: np.ndarray,
    target_name: str,
    profile: str = "balanced",
    max_nonzero: Optional[int] = 4,
    l2: float = 1.0e-4,
) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    n_models = P.shape[1]

    if n_models == 1:
        return np.ones(1)

    m = np.isfinite(y) & np.all(np.isfinite(P), axis=1)
    P_fit = P[m]
    y_fit = y[m]

    if len(y_fit) < 8:
        return np.ones(n_models) / n_models

    if minimize is None:
        scores = []
        for j in range(n_models):
            scores.append(selection_objective_value(y_fit, P_fit[:, j], target_name, profile=profile))
        scores = np.asarray(scores, dtype=float)
        inv = 1.0 / (scores + EPS)
        w = inv / np.sum(inv)
    else:
        def loss(w: np.ndarray) -> float:
            pred = P_fit @ w
            return selection_objective_value(y_fit, pred, target_name, profile=profile) + l2 * np.sum(w ** 2)

        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * n_models
        w0 = np.ones(n_models) / n_models

        res = minimize(
            loss,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 500},
        )

        w = np.asarray(res.x, dtype=float) if res.success else w0

    w = np.maximum(w, 0.0)
    w = w / np.sum(w) if np.sum(w) > EPS else np.ones(n_models) / n_models

    if max_nonzero is not None and n_models > max_nonzero:
        keep = np.argsort(w)[::-1][:max_nonzero]
        w2 = np.zeros_like(w)
        w2[keep] = w[keep]
        w = w2 / np.sum(w2) if np.sum(w2) > EPS else w

    return w


def choose_leader_names(
    all_candidates: Dict[str, Dict[str, Any]],
    candidate_names: List[str],
    y_val: np.ndarray,
    y_cal: np.ndarray,
    target_name: str,
) -> Tuple[List[str], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []

    for cname in candidate_names:
        pv = all_candidates[cname]["val"]
        pc = all_candidates[cname]["cal"]

        met_v = hydro_metrics(y_val, pv)
        met_c = hydro_metrics(y_cal, pc)

        rows.append(
            {
                "candidate": cname,
                "val_MAE": met_v["MAE"],
                "val_RMSE": met_v["RMSE"],
                "val_NSE": met_v["NSE"],
                "val_KGE": met_v["KGE"],
                "cal_MAE": met_c["MAE"],
                "cal_RMSE": met_c["RMSE"],
                "cal_NSE": met_c["NSE"],
                "cal_KGE": met_c["KGE"],
                "score_balanced": (
                    selection_objective_value(y_val, pv, target_name, "balanced")
                    + selection_objective_value(y_cal, pc, target_name, "balanced")
                ),
                "score_rmse_nse": (
                    selection_objective_value(y_val, pv, target_name, "rmse_nse")
                    + selection_objective_value(y_cal, pc, target_name, "rmse_nse")
                ),
                "score_mae": (
                    selection_objective_value(y_val, pv, target_name, "mae")
                    + selection_objective_value(y_cal, pc, target_name, "mae")
                ),
                "score_kge_bias": (
                    selection_objective_value(y_val, pv, target_name, "kge_bias")
                    + selection_objective_value(y_cal, pc, target_name, "kge_bias")
                ),
                "score_peak": (
                    selection_objective_value(y_val, pv, target_name, "peak")
                    + selection_objective_value(y_cal, pc, target_name, "peak")
                ),
            }
        )

    df = pd.DataFrame(rows)

    leader_set = set()
    for col in ["score_balanced", "score_rmse_nse", "score_mae", "score_kge_bias", "score_peak"]:
        leader_set.update(df.sort_values(col).head(8)["candidate"].tolist())

    return list(leader_set), df


def add_metric_oriented_stacks(
    all_candidates: Dict[str, Dict[str, Any]],
    candidate_names: List[str],
    y_val: np.ndarray,
    y_cal: np.ndarray,
    target_name: str,
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    leader_names, leader_df = choose_leader_names(
        all_candidates=all_candidates,
        candidate_names=candidate_names,
        y_val=y_val,
        y_cal=y_cal,
        target_name=target_name,
    )

    if len(leader_names) < 2:
        return all_candidates, leader_df

    P_train = np.column_stack([all_candidates[n]["train"] for n in leader_names])
    P_val = np.column_stack([all_candidates[n]["val"] for n in leader_names])
    P_cal = np.column_stack([all_candidates[n]["cal"] for n in leader_names])
    P_test = np.column_stack([all_candidates[n]["test"] for n in leader_names])

    P_opt = np.vstack([P_val, P_cal])
    y_opt = np.concatenate([y_val, y_cal])

    profiles = ["balanced", "rmse_nse", "mae", "kge_bias", "peak"]

    for profile in profiles:
        w = fit_simplex_weights(
            P_opt,
            y_opt,
            target_name=target_name,
            profile=profile,
            max_nonzero=STACK_MAX_NONZERO,
            l2=STACK_L2,
        )

        sname = f"MetricStack_{profile}"

        all_candidates[sname] = {
            "train": physical_projection(P_train @ w, target_name),
            "val": physical_projection(P_val @ w, target_name),
            "cal": physical_projection(P_cal @ w, target_name),
            "test": physical_projection(P_test @ w, target_name),
            "kind": f"metric_oriented_stack_{profile}",
            "model": None,
            "weights": [
                {"candidate": n, "weight": float(wi)}
                for n, wi in zip(leader_names, w)
                if wi > 1.0e-8
            ],
        }

    return all_candidates, leader_df


def adaptive_blend_grid(
    y_opt: np.ndarray,
    a_opt: np.ndarray,
    b_opt: np.ndarray,
    target_name: str,
    profile: str,
) -> Tuple[float, float]:
    best_w = 0.5
    best_obj = np.inf

    for w in np.linspace(0.0, 1.0, 51):
        p = w * a_opt + (1.0 - w) * b_opt
        p = physical_projection(p, target_name)
        obj = selection_objective_value(y_opt, p, target_name, profile=profile)

        if obj < best_obj:
            best_obj = obj
            best_w = float(w)

    return best_w, float(best_obj)


def add_adaptive_leader_blends(
    all_candidates: Dict[str, Dict[str, Any]],
    candidate_names: List[str],
    y_val: np.ndarray,
    y_cal: np.ndarray,
    target_name: str,
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    leader_names, leader_df = choose_leader_names(
        all_candidates=all_candidates,
        candidate_names=candidate_names,
        y_val=y_val,
        y_cal=y_cal,
        target_name=target_name,
    )

    if len(leader_names) < 2:
        return all_candidates, leader_df

    leader_names = leader_names[:10]
    profiles = ["balanced", "rmse_nse", "mae", "kge_bias"]

    for i in range(len(leader_names)):
        for j in range(i + 1, len(leader_names)):
            a = leader_names[i]
            b = leader_names[j]

            a_opt = np.concatenate([all_candidates[a]["val"], all_candidates[a]["cal"]])
            b_opt = np.concatenate([all_candidates[b]["val"], all_candidates[b]["cal"]])
            y_opt = np.concatenate([y_val, y_cal])

            for profile in profiles:
                w, obj = adaptive_blend_grid(
                    y_opt=y_opt,
                    a_opt=a_opt,
                    b_opt=b_opt,
                    target_name=target_name,
                    profile=profile,
                )

                if w <= 0.02 or w >= 0.98:
                    continue

                cname = f"Blend_{profile}_{a}__{b}"

                all_candidates[cname] = {
                    "train": physical_projection(
                        w * all_candidates[a]["train"] + (1.0 - w) * all_candidates[b]["train"],
                        target_name,
                    ),
                    "val": physical_projection(
                        w * all_candidates[a]["val"] + (1.0 - w) * all_candidates[b]["val"],
                        target_name,
                    ),
                    "cal": physical_projection(
                        w * all_candidates[a]["cal"] + (1.0 - w) * all_candidates[b]["cal"],
                        target_name,
                    ),
                    "test": physical_projection(
                        w * all_candidates[a]["test"] + (1.0 - w) * all_candidates[b]["test"],
                        target_name,
                    ),
                    "kind": f"adaptive_validation_calibration_blend_{profile}",
                    "model": None,
                    "source_a": a,
                    "source_b": b,
                    "weight_a": w,
                    "weight_b": 1.0 - w,
                    "blend_objective": obj,
                }

    return all_candidates, leader_df


# =============================================================================
# Robust internal candidate selection
# =============================================================================

def make_regime_masks(ref: pd.DataFrame, y: np.ndarray) -> Dict[str, np.ndarray]:
    y = np.asarray(y, dtype=float)
    masks: Dict[str, np.ndarray] = {}

    masks["all"] = np.ones_like(y, dtype=bool)

    if "P_sum_7_issue" in ref.columns:
        p7 = pd.to_numeric(ref["P_sum_7_issue"], errors="coerce").values.astype(float)
        finite = np.isfinite(p7)

        if np.sum(finite) >= 10:
            thr_dry = np.nanquantile(p7, 0.40)
            thr_wet = np.nanquantile(p7, 0.70)
            masks["dry_p7"] = finite & (p7 <= thr_dry)
            masks["wet_p7"] = finite & (p7 >= thr_wet)

    if len(y) >= 10:
        masks["low_y"] = y <= np.nanquantile(y, 0.50)
        masks["high_y"] = y >= np.nanquantile(y, 0.90)

    if "dry_p7" in masks and "low_y" in masks:
        masks["dry_low_y"] = masks["dry_p7"] & masks["low_y"]

    return masks


def candidate_complexity_score(candidate_name: str, candidate_kind: Any) -> float:
    name = str(candidate_name)
    kind = str(candidate_kind)

    score = 1.0

    if "raw_ml" in kind:
        score = 1.0
    elif "log1p_ml" in kind:
        score = 1.1
    elif "issue_state_residual" in kind:
        score = 1.2
    elif "hydrological_inertia_shrinkage" in kind:
        score = 1.4
    elif "dry_low_conditional_shrinkage" in kind:
        score = 1.5
    elif "metric_oriented_stack" in kind:
        score = 2.0
    elif "adaptive_validation_calibration_blend" in kind:
        score = 2.5

    if "Blend_" in name:
        score += 0.4
    if "MetricStack_" in name:
        score += 0.2

    return float(score)


def target_specific_selection_weights(horizon: int, target_name: str) -> Dict[str, float]:
    if target_name == "Q_point" and horizon == 1:
        return {
            "val_RMSE": 0.13,
            "val_MAE": 0.11,
            "val_NSE": 0.10,
            "val_KGE": 0.07,
            "cal_RMSE": 0.09,
            "cal_MAE": 0.07,
            "cal_NSE": 0.08,
            "cal_KGE": 0.06,
            "cal_abs_PBIAS": 0.02,
            "val_low_y_MAE": 0.06,
            "cal_low_y_MAE": 0.04,
            "val_low_y_pos_bias_ratio": 0.05,
            "cal_low_y_pos_bias_ratio": 0.03,
            "val_dry_low_MAE": 0.05,
            "cal_dry_low_MAE": 0.03,
            "val_high_MAE": 0.02,
            "cal_high_MAE": 0.01,
            "RMSE_val_cal_gap": 0.02,
            "MAE_val_cal_gap": 0.01,
            "candidate_complexity": 0.03,
        }

    if target_name == "Q_point":
        return {
            "val_RMSE": 0.16,
            "val_MAE": 0.10,
            "val_NSE": 0.13,
            "val_KGE": 0.10,
            "cal_RMSE": 0.14,
            "cal_MAE": 0.08,
            "cal_NSE": 0.11,
            "cal_KGE": 0.09,
            "cal_abs_PBIAS": 0.03,
            "val_dry_RMSE": 0.02,
            "cal_dry_RMSE": 0.02,
            "val_low_y_MAE": 0.02,
            "cal_low_y_MAE": 0.01,
            "RMSE_val_cal_gap": 0.01,
            "MAE_val_cal_gap": 0.01,
        }

    if target_name == "Q_max":
        return {
            "val_RMSE": 0.14,
            "val_MAE": 0.08,
            "val_NSE": 0.11,
            "val_KGE": 0.13,
            "cal_RMSE": 0.13,
            "cal_MAE": 0.07,
            "cal_NSE": 0.10,
            "cal_KGE": 0.12,
            "cal_abs_PBIAS": 0.03,
            "val_high_MAE": 0.04,
            "cal_high_MAE": 0.04,
            "RMSE_val_cal_gap": 0.01,
        }

    return {
        "val_RMSE": 0.18,
        "val_MAE": 0.14,
        "val_NSE": 0.14,
        "val_KGE": 0.10,
        "cal_RMSE": 0.08,
        "cal_MAE": 0.07,
        "cal_NSE": 0.07,
        "cal_KGE": 0.06,
        "cal_abs_PBIAS": 0.02,
        "val_dry_RMSE": 0.05,
        "cal_dry_RMSE": 0.03,
        "RMSE_val_cal_gap": 0.04,
        "MAE_val_cal_gap": 0.02,
    }


def robust_candidate_ranking(
    all_candidates: Dict[str, Dict[str, Any]],
    candidate_names: List[str],
    y_val: np.ndarray,
    y_cal: np.ndarray,
    target_name: str,
    horizon: int,
    ref_val: pd.DataFrame,
    ref_cal: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    calibrators: Dict[str, Dict[str, Any]] = {}

    masks_val = make_regime_masks(ref_val, y_val)
    masks_cal = make_regime_masks(ref_cal, y_cal)

    for cname in candidate_names:
        try:
            apply_cal, cal_info, cal_metrics = fit_auto_calibration(
                y_cal=y_cal,
                p_cal=all_candidates[cname]["cal"],
                target_name=target_name,
                y_val=y_val,
                p_val=all_candidates[cname]["val"],
                horizon=horizon,
            )

            pred_val = apply_cal(all_candidates[cname]["val"])
            pred_cal = apply_cal(all_candidates[cname]["cal"])

            met_val = hydro_metrics(y_val, pred_val)
            met_cal = hydro_metrics(y_cal, pred_cal)

            rmse_gap = abs(met_val["RMSE"] - met_cal["RMSE"]) / (
                0.5 * (met_val["RMSE"] + met_cal["RMSE"]) + EPS
            )

            mae_gap = abs(met_val["MAE"] - met_cal["MAE"]) / (
                0.5 * (met_val["MAE"] + met_cal["MAE"]) + EPS
            )

            kind = all_candidates[cname].get("kind", None)

            row = {
                "candidate": cname,
                "candidate_kind": kind,
                "candidate_complexity": candidate_complexity_score(cname, kind),
                "calibration_info": json_dumps(cal_info),
                "val_MAE": met_val["MAE"],
                "val_RMSE": met_val["RMSE"],
                "val_NSE": met_val["NSE"],
                "val_KGE": met_val["KGE"],
                "val_Bias": met_val["Bias"],
                "val_PBIAS_percent": met_val["PBIAS_percent"],
                "cal_MAE": met_cal["MAE"],
                "cal_RMSE": met_cal["RMSE"],
                "cal_NSE": met_cal["NSE"],
                "cal_KGE": met_cal["KGE"],
                "cal_Bias": met_cal["Bias"],
                "cal_PBIAS_percent": met_cal["PBIAS_percent"],
                "cal_abs_PBIAS": abs(met_cal["PBIAS_percent"]),
                "RMSE_val_cal_gap": rmse_gap,
                "MAE_val_cal_gap": mae_gap,
            }

            row["val_dry_RMSE"] = masked_metric(
                y_val,
                pred_val,
                masks_val.get("dry_p7", masks_val["all"]),
                "RMSE",
            )

            row["cal_dry_RMSE"] = masked_metric(
                y_cal,
                pred_cal,
                masks_cal.get("dry_p7", masks_cal["all"]),
                "RMSE",
            )

            row["val_high_MAE"] = masked_mae(
                y_val,
                pred_val,
                masks_val.get("high_y", masks_val["all"]),
            )

            row["cal_high_MAE"] = masked_mae(
                y_cal,
                pred_cal,
                masks_cal.get("high_y", masks_cal["all"]),
            )

            row["val_low_y_MAE"] = masked_mae(
                y_val,
                pred_val,
                masks_val.get("low_y", masks_val["all"]),
            )

            row["cal_low_y_MAE"] = masked_mae(
                y_cal,
                pred_cal,
                masks_cal.get("low_y", masks_cal["all"]),
            )

            row["val_low_y_pos_bias_ratio"] = masked_positive_bias_ratio(
                y_val,
                pred_val,
                masks_val.get("low_y", masks_val["all"]),
            )

            row["cal_low_y_pos_bias_ratio"] = masked_positive_bias_ratio(
                y_cal,
                pred_cal,
                masks_cal.get("low_y", masks_cal["all"]),
            )

            row["val_dry_low_MAE"] = masked_mae(
                y_val,
                pred_val,
                masks_val.get("dry_low_y", masks_val.get("low_y", masks_val["all"])),
            )

            row["cal_dry_low_MAE"] = masked_mae(
                y_cal,
                pred_cal,
                masks_cal.get("dry_low_y", masks_cal.get("low_y", masks_cal["all"])),
            )

            rows.append(row)

            calibrators[cname] = {
                "apply": apply_cal,
                "info": cal_info,
                "calibration_candidate_metrics": cal_metrics,
            }

        except Exception as exc:
            rows.append(
                {
                    "candidate": cname,
                    "candidate_kind": all_candidates[cname].get("kind", None),
                    "error": str(exc),
                }
            )

    df = pd.DataFrame(rows)

    weights = target_specific_selection_weights(horizon, target_name)

    for metric in weights.keys():
        if metric not in df.columns:
            df[metric] = np.nan

        ascending = True
        if metric.endswith("NSE") or metric.endswith("KGE"):
            ascending = False

        df[f"rank_{metric}"] = df[metric].rank(
            method="average",
            ascending=ascending,
            na_option="bottom",
        )

    df["robust_selection_score"] = 0.0

    for metric, weight in weights.items():
        df["robust_selection_score"] += weight * df[f"rank_{metric}"]

    df["selection_profile"] = json_dumps(weights)
    df = df.sort_values("robust_selection_score").reset_index(drop=True)

    return df, calibrators


# =============================================================================
# Training the proposed model for one target
# =============================================================================

def train_target_rs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_name: str,
    horizon: int,
    issue_state_train: np.ndarray,
    issue_state_val: np.ndarray,
    issue_state_cal: np.ndarray,
    issue_state_test: np.ndarray,
    ref_train: pd.DataFrame,
    ref_val: pd.DataFrame,
    ref_cal: pd.DataFrame,
    ref_test: pd.DataFrame,
    random_state: int = 42,
) -> Dict[str, Any]:
    target_type = "Q" if target_name.upper().startswith("Q") else "H"

    internal_learners = build_internal_learners(random_state=random_state)
    sample_weight = sample_weight_for_target(y_train, target_name)

    all_candidates: Dict[str, Dict[str, Any]] = {}

    for name, model in internal_learners.items():
        print(f"      Fitting internal raw learner: {name}")

        try:
            m = fit_model(model, X_train, y_train, sample_weight)

            all_candidates[f"{name}_Raw"] = {
                "train": physical_projection(predict_1d(m, X_train), target_name),
                "val": physical_projection(predict_1d(m, X_val), target_name),
                "cal": physical_projection(predict_1d(m, X_cal), target_name),
                "test": physical_projection(predict_1d(m, X_test), target_name),
                "kind": "raw_ml_internal_candidate",
                "model": m,
            }

        except Exception as exc:
            print(f"        Warning: internal learner {name} failed: {exc}")

    if target_type == "Q":
        y_train_log = np.log1p(np.maximum(y_train, 0.0))

        for name, model in internal_learners.items():
            print(f"      Fitting internal log1p learner: {name}")

            try:
                m = fit_model(model, X_train, y_train_log, sample_weight)

                all_candidates[f"{name}_Log1p"] = {
                    "train": physical_projection(safe_expm1(predict_1d(m, X_train)), target_name),
                    "val": physical_projection(safe_expm1(predict_1d(m, X_val)), target_name),
                    "cal": physical_projection(safe_expm1(predict_1d(m, X_cal)), target_name),
                    "test": physical_projection(safe_expm1(predict_1d(m, X_test)), target_name),
                    "kind": "log1p_ml_internal_candidate",
                    "model": m,
                }

            except Exception as exc:
                print(f"        Warning: internal log1p learner {name} failed: {exc}")

    issue_state_train = np.asarray(issue_state_train, dtype=float)
    issue_state_val = np.asarray(issue_state_val, dtype=float)
    issue_state_cal = np.asarray(issue_state_cal, dtype=float)
    issue_state_test = np.asarray(issue_state_test, dtype=float)

    if target_type == "Q":
        y_res_train = (
            np.log1p(np.maximum(y_train, 0.0))
            - np.log1p(np.maximum(issue_state_train, 0.0))
        )
    else:
        y_res_train = y_train - issue_state_train

    for name, model in internal_learners.items():
        print(f"      Fitting internal issue-state residual learner: {name}")

        try:
            m = fit_model(model, X_train, y_res_train, sample_weight)

            res_train = predict_1d(m, X_train)
            res_val = predict_1d(m, X_val)
            res_cal = predict_1d(m, X_cal)
            res_test = predict_1d(m, X_test)

            if target_type == "Q":
                pred_train = safe_expm1(np.log1p(np.maximum(issue_state_train, 0.0)) + res_train)
                pred_val = safe_expm1(np.log1p(np.maximum(issue_state_val, 0.0)) + res_val)
                pred_cal = safe_expm1(np.log1p(np.maximum(issue_state_cal, 0.0)) + res_cal)
                pred_test = safe_expm1(np.log1p(np.maximum(issue_state_test, 0.0)) + res_test)
            else:
                pred_train = issue_state_train + res_train
                pred_val = issue_state_val + res_val
                pred_cal = issue_state_cal + res_cal
                pred_test = issue_state_test + res_test

            all_candidates[f"{name}_StateResidual"] = {
                "train": physical_projection(pred_train, target_name),
                "val": physical_projection(pred_val, target_name),
                "cal": physical_projection(pred_cal, target_name),
                "test": physical_projection(pred_test, target_name),
                "kind": "issue_state_residual_internal_candidate",
                "model": m,
            }

        except Exception as exc:
            print(f"        Warning: internal residual learner {name} failed: {exc}")

    if ADD_INERTIA_SHRINKAGE_CANDIDATES:
        source_names = list(all_candidates.keys())
        print(f"      Adding inertia-shrinkage candidates from {len(source_names)} internal sources")

        for cname in source_names:
            try:
                lam, obj = optimize_shrinkage_lambda(
                    y_val=y_val,
                    base_val=all_candidates[cname]["val"],
                    state_val=issue_state_val,
                    target_name=target_name,
                )

                sname = f"{cname}_InertiaShrink"

                all_candidates[sname] = {
                    "train": apply_inertia_shrinkage(all_candidates[cname]["train"], issue_state_train, target_name, lam),
                    "val": apply_inertia_shrinkage(all_candidates[cname]["val"], issue_state_val, target_name, lam),
                    "cal": apply_inertia_shrinkage(all_candidates[cname]["cal"], issue_state_cal, target_name, lam),
                    "test": apply_inertia_shrinkage(all_candidates[cname]["test"], issue_state_test, target_name, lam),
                    "kind": "hydrological_inertia_shrinkage",
                    "model": all_candidates[cname].get("model", None),
                    "source_candidate": cname,
                    "shrinkage_lambda": lam,
                    "shrinkage_val_objective": obj,
                }

            except Exception as exc:
                print(f"        Warning: inertia-shrinkage candidate from {cname} failed: {exc}")

    if ADD_DRY_LOW_SHRINKAGE_CANDIDATES and target_type in DRY_LOW_SHRINKAGE_TARGET_TYPES:
        dry_info = dry_low_masks_from_refs(
            ref_train=ref_train,
            ref_val=ref_val,
            ref_cal=ref_cal,
            ref_test=ref_test,
            state_train=issue_state_train,
            state_val=issue_state_val,
            state_cal=issue_state_cal,
            state_test=issue_state_test,
            target_name=target_name,
        )

        if dry_info is not None:
            source_names = list(all_candidates.keys())
            print(f"      Adding dry-low conditional shrinkage candidates from {len(source_names)} internal sources")

            for cname in source_names:
                try:
                    lam, obj = optimize_dry_low_shrinkage_lambda(
                        y_val=y_val,
                        y_cal=y_cal,
                        base_val=all_candidates[cname]["val"],
                        base_cal=all_candidates[cname]["cal"],
                        state_val=issue_state_val,
                        state_cal=issue_state_cal,
                        target_name=target_name,
                        mask_val=dry_info["val"],
                        mask_cal=dry_info["cal"],
                        horizon=horizon,
                    )

                    if lam is None:
                        continue

                    sname = f"{cname}_DryLowShrink"

                    all_candidates[sname] = {
                        "train": apply_conditional_dry_low_shrinkage(
                            all_candidates[cname]["train"],
                            issue_state_train,
                            target_name,
                            dry_info["train"],
                            lam,
                        ),
                        "val": apply_conditional_dry_low_shrinkage(
                            all_candidates[cname]["val"],
                            issue_state_val,
                            target_name,
                            dry_info["val"],
                            lam,
                        ),
                        "cal": apply_conditional_dry_low_shrinkage(
                            all_candidates[cname]["cal"],
                            issue_state_cal,
                            target_name,
                            dry_info["cal"],
                            lam,
                        ),
                        "test": apply_conditional_dry_low_shrinkage(
                            all_candidates[cname]["test"],
                            issue_state_test,
                            target_name,
                            dry_info["test"],
                            lam,
                        ),
                        "kind": "dry_low_conditional_shrinkage",
                        "model": all_candidates[cname].get("model", None),
                        "source_candidate": cname,
                        "dry_low_lambda": lam,
                        "dry_low_objective": obj,
                        "dry_low_p7_threshold": dry_info["p7_threshold"],
                        "dry_low_state_threshold": dry_info["state_threshold"],
                        "dry_low_val_count": int(np.sum(dry_info["val"])),
                        "dry_low_cal_count": int(np.sum(dry_info["cal"])),
                    }

                except Exception as exc:
                    print(f"        Warning: dry-low candidate from {cname} failed: {exc}")

    proposed_candidate_names = list(all_candidates.keys())

    if len(proposed_candidate_names) == 0:
        raise RuntimeError("No valid internal candidate is available for the proposed model.")

    leader_df = pd.DataFrame()

    if ADD_METRIC_ORIENTED_STACKS:
        print("      Adding metric-oriented stack candidates")

        all_candidates, leader_df_stack = add_metric_oriented_stacks(
            all_candidates=all_candidates,
            candidate_names=proposed_candidate_names,
            y_val=y_val,
            y_cal=y_cal,
            target_name=target_name,
        )

        leader_df = leader_df_stack.copy()
        proposed_candidate_names = list(all_candidates.keys())

    if ADD_ADAPTIVE_LEADER_BLENDS:
        print("      Adding adaptive leader blend candidates")

        all_candidates, leader_df_blend = add_adaptive_leader_blends(
            all_candidates=all_candidates,
            candidate_names=proposed_candidate_names,
            y_val=y_val,
            y_cal=y_cal,
            target_name=target_name,
        )

        if leader_df.empty:
            leader_df = leader_df_blend.copy()

        proposed_candidate_names = list(all_candidates.keys())

    robust_ranked, candidate_calibrators = robust_candidate_ranking(
        all_candidates=all_candidates,
        candidate_names=proposed_candidate_names,
        y_val=y_val,
        y_cal=y_cal,
        target_name=target_name,
        horizon=horizon,
        ref_val=ref_val,
        ref_cal=ref_cal,
    )

    valid_ranked = robust_ranked[robust_ranked["candidate"].isin(candidate_calibrators.keys())].copy()
    if valid_ranked.empty:
        raise RuntimeError(f"No valid calibrated internal candidate for target {target_name}.")

    selected_name = str(valid_ranked.iloc[0]["candidate"])
    selected_obj = all_candidates[selected_name]
    selected_apply = candidate_calibrators[selected_name]["apply"]
    selected_cal_info = candidate_calibrators[selected_name]["info"]
    selected_cal_metrics = candidate_calibrators[selected_name]["calibration_candidate_metrics"]

    pred_train_final = selected_apply(selected_obj["train"])
    pred_val_final = selected_apply(selected_obj["val"])
    pred_cal_final = selected_apply(selected_obj["cal"])
    pred_test_final = selected_apply(selected_obj["test"])

    active_weights = []
    if selected_obj.get("weights") is not None:
        active_weights = selected_obj.get("weights")

    return {
        "selected_candidate": selected_name,
        "selected_val_metrics": valid_ranked.iloc[0].to_dict(),
        "internal_candidate_ranking": robust_ranked,
        "leader_candidate_summary": leader_df,
        "stack_active_weights": active_weights,
        "selected_object": selected_obj,
        "calibration_method": "safe_auto_selected_on_validation_plus_calibration",
        "calibration_info": selected_cal_info,
        "calibration_candidate_metrics": selected_cal_metrics,
        "y_pred_train": pred_train_final,
        "y_pred_val": pred_val_final,
        "y_pred_cal": pred_cal_final,
        "y_pred_test": pred_test_final,
    }


# =============================================================================
# Q-H physical direction consistency
# =============================================================================

def qh_direction_inconsistency(
    q_pred: np.ndarray,
    h_pred: np.ndarray,
    q_state: np.ndarray,
    h_state: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    dq = np.asarray(q_pred, dtype=float) - np.asarray(q_state, dtype=float)
    dh = np.asarray(h_pred, dtype=float) - np.asarray(h_state, dtype=float)

    valid = np.isfinite(dq) & np.isfinite(dh)
    inconsistent = valid & (dq * dh < 0.0)

    return inconsistent, valid


def apply_qh_direction_constraint_to_split(
    q_pred: np.ndarray,
    h_pred: np.ndarray,
    q_state: np.ndarray,
    h_state: np.ndarray,
    q_scale: float,
    h_scale: float,
):
    q_pred = np.asarray(q_pred, dtype=float).copy()
    h_pred = np.asarray(h_pred, dtype=float).copy()

    q_state = np.asarray(q_state, dtype=float)
    h_state = np.asarray(h_state, dtype=float)

    before_incon, valid = qh_direction_inconsistency(q_pred, h_pred, q_state, h_state)

    dq = q_pred - q_state
    dh = h_pred - h_state

    q_mag = np.abs(dq) / (q_scale + EPS)
    h_mag = np.abs(dh) / (h_scale + EPS)

    idx = np.where(before_incon)[0]

    q_adjusted_count = 0
    h_adjusted_count = 0

    for i in idx:
        if q_mag[i] <= h_mag[i]:
            q_pred[i] = q_state[i]
            q_adjusted_count += 1
        else:
            h_pred[i] = h_state[i]
            h_adjusted_count += 1

    q_pred = np.maximum(q_pred, 0.0)

    after_incon, valid_after = qh_direction_inconsistency(q_pred, h_pred, q_state, h_state)

    stats = {
        "N": int(np.sum(valid)),
        "inconsistent_before_count": int(np.sum(before_incon)),
        "inconsistent_before_rate": float(np.mean(before_incon[valid])) if np.sum(valid) > 0 else np.nan,
        "inconsistent_after_count": int(np.sum(after_incon)),
        "inconsistent_after_rate": float(np.mean(after_incon[valid_after])) if np.sum(valid_after) > 0 else np.nan,
        "q_adjusted_count": int(q_adjusted_count),
        "h_adjusted_count": int(h_adjusted_count),
    }

    return q_pred, h_pred, stats


def joint_qh_validation_calibration_score(
    yq_val: np.ndarray,
    yh_val: np.ndarray,
    pq_val: np.ndarray,
    ph_val: np.ndarray,
    yq_cal: np.ndarray,
    yh_cal: np.ndarray,
    pq_cal: np.ndarray,
    ph_cal: np.ndarray,
) -> float:
    yq = np.concatenate([np.asarray(yq_val, dtype=float), np.asarray(yq_cal, dtype=float)])
    yh = np.concatenate([np.asarray(yh_val, dtype=float), np.asarray(yh_cal, dtype=float)])

    pq = np.concatenate([np.asarray(pq_val, dtype=float), np.asarray(pq_cal, dtype=float)])
    ph = np.concatenate([np.asarray(ph_val, dtype=float), np.asarray(ph_cal, dtype=float)])

    q_scale = np.nanstd(yq) + EPS
    h_scale = np.nanstd(yh) + EPS

    q_obj = selection_objective_value(yq, pq, "Q_point", profile="balanced") / q_scale
    h_obj = selection_objective_value(yh, ph, "H_point", profile="balanced") / h_scale

    return float(0.50 * q_obj + 0.50 * h_obj)


def should_apply_qh_direction_constraint(
    horizon: int,
    result_q: Dict[str, Any],
    result_h: Dict[str, Any],
    data_q: Dict[str, np.ndarray],
    data_h: Dict[str, np.ndarray],
):
    q_cal = np.asarray(data_q["y_cal"], dtype=float)
    h_cal = np.asarray(data_h["y_cal"], dtype=float)

    q_state_cal = np.asarray(data_q["state_cal"], dtype=float)
    h_state_cal = np.asarray(data_h["state_cal"], dtype=float)

    q_scale = float(np.nanmedian(np.abs(q_cal - q_state_cal))) + EPS
    h_scale = float(np.nanmedian(np.abs(h_cal - h_state_cal))) + EPS

    before_score = joint_qh_validation_calibration_score(
        yq_val=data_q["y_val"],
        yh_val=data_h["y_val"],
        pq_val=result_q["y_pred_val"],
        ph_val=result_h["y_pred_val"],
        yq_cal=data_q["y_cal"],
        yh_cal=data_h["y_cal"],
        pq_cal=result_q["y_pred_cal"],
        ph_cal=result_h["y_pred_cal"],
    )

    q_val_adj, h_val_adj, _ = apply_qh_direction_constraint_to_split(
        q_pred=result_q["y_pred_val"],
        h_pred=result_h["y_pred_val"],
        q_state=data_q["state_val"],
        h_state=data_h["state_val"],
        q_scale=q_scale,
        h_scale=h_scale,
    )

    q_cal_adj, h_cal_adj, _ = apply_qh_direction_constraint_to_split(
        q_pred=result_q["y_pred_cal"],
        h_pred=result_h["y_pred_cal"],
        q_state=data_q["state_cal"],
        h_state=data_h["state_cal"],
        q_scale=q_scale,
        h_scale=h_scale,
    )

    after_score = joint_qh_validation_calibration_score(
        yq_val=data_q["y_val"],
        yh_val=data_h["y_val"],
        pq_val=q_val_adj,
        ph_val=h_val_adj,
        yq_cal=data_q["y_cal"],
        yh_cal=data_h["y_cal"],
        pq_cal=q_cal_adj,
        ph_cal=h_cal_adj,
    )

    apply_flag = after_score <= before_score * (1.0 + QH_CONSTRAINT_MAX_RELATIVE_DEGRADATION)

    return bool(apply_flag), {
        "horizon": horizon,
        "before_joint_val_cal_score": before_score,
        "after_joint_val_cal_score": after_score,
        "relative_change": (after_score - before_score) / (abs(before_score) + EPS),
        "adaptive_qh_constraint_applied": bool(apply_flag),
        "max_allowed_relative_degradation": QH_CONSTRAINT_MAX_RELATIVE_DEGRADATION,
    }


def apply_qh_direction_constraint_to_results(
    horizon: int,
    result_q: Dict[str, Any],
    result_h: Dict[str, Any],
    data_q: Dict[str, np.ndarray],
    data_h: Dict[str, np.ndarray],
):
    q_cal = np.asarray(data_q["y_cal"], dtype=float)
    h_cal = np.asarray(data_h["y_cal"], dtype=float)

    q_state_cal = np.asarray(data_q["state_cal"], dtype=float)
    h_state_cal = np.asarray(data_h["state_cal"], dtype=float)

    q_scale = float(np.nanmedian(np.abs(q_cal - q_state_cal))) + EPS
    h_scale = float(np.nanmedian(np.abs(h_cal - h_state_cal))) + EPS

    rows: List[Dict[str, Any]] = []

    for split_name in ["train", "val", "cal", "test"]:
        q_key = f"y_pred_{split_name}"
        h_key = f"y_pred_{split_name}"

        q_adj, h_adj, stats = apply_qh_direction_constraint_to_split(
            q_pred=result_q[q_key],
            h_pred=result_h[h_key],
            q_state=data_q[f"state_{split_name}"],
            h_state=data_h[f"state_{split_name}"],
            q_scale=q_scale,
            h_scale=h_scale,
        )

        result_q[q_key] = q_adj
        result_h[h_key] = h_adj

        rows.append(
            {
                "horizon": horizon,
                "split": split_name,
                "q_scale_from_calibration": q_scale,
                "h_scale_from_calibration": h_scale,
                **stats,
            }
        )

    return result_q, result_h, pd.DataFrame(rows)


def physical_consistency_audit(
    horizon: int,
    ref_test: pd.DataFrame,
    pred_Q_point: np.ndarray,
    pred_H_point: np.ndarray,
) -> pd.DataFrame:
    q_state = ref_test["issue_state_Q"].values.astype(float)
    h_state = ref_test["issue_state_H"].values.astype(float)

    dq_pred = np.asarray(pred_Q_point, dtype=float) - q_state
    dh_pred = np.asarray(pred_H_point, dtype=float) - h_state

    dq_obs = ref_test["Observed_Q_point"].values.astype(float) - q_state
    dh_obs = ref_test["Observed_H_point"].values.astype(float) - h_state

    valid_pred = np.isfinite(dq_pred) & np.isfinite(dh_pred)
    valid_obs = np.isfinite(dq_obs) & np.isfinite(dh_obs)

    pred_inconsistent = valid_pred & (dq_pred * dh_pred < 0.0)
    obs_inconsistent = valid_obs & (dq_obs * dh_obs < 0.0)

    return pd.DataFrame(
        [
            {
                "horizon": horizon,
                "model": PROPOSED_NAME,
                "N": int(np.sum(valid_pred)),
                "pred_QH_direction_inconsistent_count": int(np.sum(pred_inconsistent)),
                "pred_QH_direction_inconsistent_rate": (
                    float(np.mean(pred_inconsistent[valid_pred])) if np.sum(valid_pred) > 0 else np.nan
                ),
                "observed_QH_direction_inconsistent_count": int(np.sum(obs_inconsistent)),
                "observed_QH_direction_inconsistent_rate": (
                    float(np.mean(obs_inconsistent[valid_obs])) if np.sum(valid_obs) > 0 else np.nan
                ),
                "negative_Q_prediction_count": int(np.sum(np.asarray(pred_Q_point) < 0.0)),
            }
        ]
    )


# =============================================================================
# Output builders
# =============================================================================

def conformal_interval_summary(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    alpha: float = 0.10,
    ref_cal: Optional[pd.DataFrame] = None,
    ref_test: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    y_cal = np.asarray(y_cal, dtype=float)
    p_cal = np.asarray(p_cal, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    p_test = np.asarray(p_test, dtype=float)

    mcal = np.isfinite(y_cal) & np.isfinite(p_cal)
    residual = np.abs(y_cal[mcal] - p_cal[mcal])

    if len(residual) < 5:
        q = np.nan
        lower = np.full_like(p_test, np.nan, dtype=float)
        upper = np.full_like(p_test, np.nan, dtype=float)
    else:
        q = float(np.quantile(residual, 1.0 - alpha))
        lower = p_test - q
        upper = p_test + q

    if np.isfinite(q):
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        mean_width = np.mean(upper - lower)
    else:
        coverage = np.nan
        mean_width = np.nan

    out: Dict[str, Any] = {
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "standard_abs_residual_quantile": q,
        "standard_test_empirical_coverage": float(coverage) if np.isfinite(coverage) else np.nan,
        "standard_test_mean_interval_width": float(mean_width) if np.isfinite(mean_width) else np.nan,
        "stratified_conformal_used": False,
        "stratified_by": None,
        "stratified_threshold": np.nan,
        "stratified_test_empirical_coverage": np.nan,
        "stratified_test_mean_interval_width": np.nan,
        "stratified_dry_quantile": np.nan,
        "stratified_wet_quantile": np.nan,
        "stratified_dry_cal_count": np.nan,
        "stratified_wet_cal_count": np.nan,
    }

    if (
        ref_cal is not None
        and ref_test is not None
        and "P_sum_7_issue" in ref_cal.columns
        and "P_sum_7_issue" in ref_test.columns
    ):
        p7_cal = pd.to_numeric(ref_cal["P_sum_7_issue"], errors="coerce").values.astype(float)
        p7_test = pd.to_numeric(ref_test["P_sum_7_issue"], errors="coerce").values.astype(float)

        finite_cal = np.isfinite(p7_cal) & np.isfinite(y_cal) & np.isfinite(p_cal)

        if np.sum(finite_cal) >= 20:
            thr = float(np.nanmedian(p7_cal[finite_cal]))

            dry_cal = finite_cal & (p7_cal <= thr)
            wet_cal = finite_cal & (p7_cal > thr)

            if np.sum(dry_cal) >= 8 and np.sum(wet_cal) >= 8 and np.isfinite(q):
                res_dry = np.abs(y_cal[dry_cal] - p_cal[dry_cal])
                res_wet = np.abs(y_cal[wet_cal] - p_cal[wet_cal])

                q_dry = float(np.quantile(res_dry, 1.0 - alpha))
                q_wet = float(np.quantile(res_wet, 1.0 - alpha))

                q_test = np.where(p7_test <= thr, q_dry, q_wet)

                lower_s = p_test - q_test
                upper_s = p_test + q_test

                valid_test = np.isfinite(y_test) & np.isfinite(p_test) & np.isfinite(q_test)

                if np.sum(valid_test) > 0:
                    cov_s = float(np.mean((y_test[valid_test] >= lower_s[valid_test]) & (y_test[valid_test] <= upper_s[valid_test])))
                    width_s = float(np.mean(upper_s[valid_test] - lower_s[valid_test]))

                    out.update(
                        {
                            "stratified_conformal_used": True,
                            "stratified_by": "P_sum_7_issue_median_from_calibration",
                            "stratified_threshold": thr,
                            "stratified_test_empirical_coverage": cov_s,
                            "stratified_test_mean_interval_width": width_s,
                            "stratified_dry_quantile": q_dry,
                            "stratified_wet_quantile": q_wet,
                            "stratified_dry_cal_count": int(np.sum(dry_cal)),
                            "stratified_wet_cal_count": int(np.sum(wet_cal)),
                        }
                    )

    return out


def build_test_metrics(
    result: Dict[str, Any],
    y_test: np.ndarray,
    horizon: int,
    target_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "horizon": horizon,
                "target": target_name,
                "model": PROPOSED_NAME,
                "selected_candidate": result["selected_candidate"],
                "prediction_type": "HydroPIML_RS_prediction",
                **hydro_metrics(y_test, result["y_pred_test"]),
            }
        ]
    )


def build_selected_summary(
    horizon: int,
    target_name: str,
    result: Dict[str, Any],
) -> pd.DataFrame:
    selected_val = result["selected_val_metrics"]

    return pd.DataFrame(
        [
            {
                "horizon": horizon,
                "target": target_name,
                "selected_candidate": result["selected_candidate"],
                "selection_objective": "target_specific_validation_plus_calibration_selection",
                "robust_selection_score": selected_val.get("robust_selection_score", np.nan),
                "selection_profile": selected_val.get("selection_profile", None),
                "val_MAE": selected_val.get("val_MAE", np.nan),
                "val_RMSE": selected_val.get("val_RMSE", np.nan),
                "val_NSE": selected_val.get("val_NSE", np.nan),
                "val_KGE": selected_val.get("val_KGE", np.nan),
                "cal_MAE": selected_val.get("cal_MAE", np.nan),
                "cal_RMSE": selected_val.get("cal_RMSE", np.nan),
                "cal_NSE": selected_val.get("cal_NSE", np.nan),
                "cal_KGE": selected_val.get("cal_KGE", np.nan),
                "cal_abs_PBIAS": selected_val.get("cal_abs_PBIAS", np.nan),
                "RMSE_val_cal_gap": selected_val.get("RMSE_val_cal_gap", np.nan),
                "MAE_val_cal_gap": selected_val.get("MAE_val_cal_gap", np.nan),
                "candidate_complexity": selected_val.get("candidate_complexity", np.nan),
                "candidate_kind": result["selected_object"].get("kind", None),
                "calibration_method": result["calibration_method"],
                "calibration_info": json_dumps(result["calibration_info"]),
            }
        ]
    )


def build_regime_metrics(
    horizon: int,
    target_name: str,
    y_test: np.ndarray,
    p_test: np.ndarray,
    state_test: np.ndarray,
    ref_cal: pd.DataFrame,
    ref_test: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    y_test = np.asarray(y_test, dtype=float)
    p_test = np.asarray(p_test, dtype=float)
    state_test = np.asarray(state_test, dtype=float)

    regimes: Dict[str, np.ndarray] = {}
    regimes["all"] = np.ones_like(y_test, dtype=bool)

    if "P_sum_7_issue" in ref_test.columns and "P_sum_7_issue" in ref_cal.columns:
        thr_p7 = np.nanmedian(ref_cal["P_sum_7_issue"].values.astype(float))
        p7_test = ref_test["P_sum_7_issue"].values.astype(float)
        regimes["dry_by_cal_Psum7_median"] = p7_test <= thr_p7
        regimes["wet_by_cal_Psum7_median"] = p7_test > thr_p7

    y_thr_low = np.nanquantile(y_test, 0.50)
    y_thr_high = np.nanquantile(y_test, 0.90)

    regimes["low_observed_le_test_median"] = y_test <= y_thr_low
    regimes["high_observed_ge_test_p90"] = y_test >= y_thr_high

    dy = y_test - state_test
    regimes["observed_rising"] = dy > 0
    regimes["observed_nonrising"] = dy <= 0

    for regime_name, mask in regimes.items():
        mask = np.asarray(mask, dtype=bool)

        if np.sum(mask) < 5:
            continue

        met = hydro_metrics(y_test[mask], p_test[mask])

        rows.append(
            {
                "horizon": horizon,
                "target": target_name,
                "model": PROPOSED_NAME,
                "regime": regime_name,
                "N_regime": int(np.sum(mask)),
                **met,
            }
        )

    return pd.DataFrame(rows)


def save_prediction_csv(
    pred_dir: Path,
    horizon: int,
    target_name: str,
    split_name: str,
    pred_dates: pd.Series,
    issue_dates: pd.Series,
    y_obs: np.ndarray,
    result: Dict[str, Any],
) -> Path:
    df = pd.DataFrame(
        {
            "issue_date": issue_dates,
            "pred_date": pred_dates,
            "Observed": y_obs,
            PROPOSED_NAME: result[f"y_pred_{split_name}"],
            "selected_candidate": result["selected_candidate"],
        }
    )

    if split_name == "test":
        path = pred_dir / f"h{horizon}_{target_name}_predictions.csv"
    else:
        path = pred_dir / f"h{horizon}_{target_name}_{split_name}_predictions.csv"

    df.to_csv(path, index=False, encoding="utf-8")

    return path


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    t0 = time.time()

    out_dir = ensure_dir(OUTPUT_DIR)
    pred_dir = ensure_dir(out_dir / "predictions")
    ref_dir = ensure_dir(out_dir / "reference")

    print_header("HydroPIML-RS prediction experiment")
    print(f"Input Excel : {INPUT_EXCEL}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"Experiment  : {EXPERIMENT_TAG}")
    print(f"Horizons    : {HORIZONS}")
    print("Reporting   : HydroPIML-RS predictions, diagnostics, and performance metrics.")
    print("Test policy : independent test period is used only for final evaluation.")

    print_section("Optional internal learner status")
    for k, v in OPTIONAL_MODEL_STATUS.items():
        print(f"{k}: {v}")

    daily, selected_cols = load_daily_data(INPUT_EXCEL)

    print_section("Daily data")
    print(f"Rows       : {len(daily)}")
    print(f"Date range : {daily['Date'].min()} -> {daily['Date'].max()}")
    print(f"Selected columns: {selected_cols}")

    feature_table, feature_cols = construct_feature_table(daily)

    print_section("Feature table")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Maximum lag/window: {max(max(LAGS), max(ROLL_WINDOWS))} days")

    all_test_metrics: List[pd.DataFrame] = []
    all_selected: List[pd.DataFrame] = []
    all_candidate_rankings: List[pd.DataFrame] = []
    all_calibration_candidates: List[pd.DataFrame] = []
    all_conformal: List[pd.DataFrame] = []
    all_regime_metrics: List[pd.DataFrame] = []
    all_physical_audit: List[pd.DataFrame] = []
    all_qh_constraint: List[pd.DataFrame] = []
    all_stack_weights: List[pd.DataFrame] = []
    all_split_diagnostics: List[pd.DataFrame] = []
    split_rows: List[Dict[str, Any]] = []

    for horizon in HORIZONS:
        supervised = make_supervised_dataset(daily, feature_table, feature_cols, horizon)
        idx_train, idx_val, idx_cal, idx_test = time_ordered_split4(len(supervised))

        ref_train = build_reference_split(supervised, idx_train)
        ref_val = build_reference_split(supervised, idx_val)
        ref_cal = build_reference_split(supervised, idx_cal)
        ref_test = build_reference_split(supervised, idx_test)

        if WRITE_REFERENCE_SPLITS:
            ref_train.to_csv(ref_dir / f"h{horizon}_train_reference.csv", index=False, encoding="utf-8")
            ref_val.to_csv(ref_dir / f"h{horizon}_val_reference.csv", index=False, encoding="utf-8")
            ref_cal.to_csv(ref_dir / f"h{horizon}_cal_reference.csv", index=False, encoding="utf-8")
            ref_test.to_csv(ref_dir / f"h{horizon}_test_reference.csv", index=False, encoding="utf-8")

        all_split_diagnostics.append(split_distribution_diagnostics(horizon, "train", ref_train))
        all_split_diagnostics.append(split_distribution_diagnostics(horizon, "validation", ref_val))
        all_split_diagnostics.append(split_distribution_diagnostics(horizon, "calibration", ref_cal))
        all_split_diagnostics.append(split_distribution_diagnostics(horizon, "test", ref_test))

        split_rows.append(
            {
                "horizon": horizon,
                "All": len(supervised),
                "Train": len(idx_train),
                "Validation": len(idx_val),
                "Calibration": len(idx_cal),
                "Test": len(idx_test),
                "train_start": ref_train["pred_date"].min(),
                "train_end": ref_train["pred_date"].max(),
                "val_start": ref_val["pred_date"].min(),
                "val_end": ref_val["pred_date"].max(),
                "cal_start": ref_cal["pred_date"].min(),
                "cal_end": ref_cal["pred_date"].max(),
                "test_start": ref_test["pred_date"].min(),
                "test_end": ref_test["pred_date"].max(),
            }
        )

        print_section(f"h={horizon} chronological split")
        print(pd.DataFrame([split_rows[-1]]).to_string(index=False))

        X = supervised[feature_cols].values.astype(float)

        X_train = X[idx_train]
        X_val = X[idx_val]
        X_cal = X[idx_cal]
        X_test = X[idx_test]

        target_names = target_list_for_horizon(horizon)
        print(f"Targets for h={horizon}: {target_names}")

        horizon_results: Dict[str, Dict[str, Any]] = {}

        for target_name in target_names:
            print_section(f"Training proposed model: h={horizon}, target={target_name}")

            y_col, state_col = target_columns(target_name)

            y_all = supervised[y_col].values.astype(float)
            state_all = supervised[state_col].values.astype(float)

            y_train = y_all[idx_train]
            y_val = y_all[idx_val]
            y_cal = y_all[idx_cal]
            y_test = y_all[idx_test]

            state_train = state_all[idx_train]
            state_val = state_all[idx_val]
            state_cal = state_all[idx_cal]
            state_test = state_all[idx_test]

            result = train_target_rs(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_cal=X_cal,
                y_cal=y_cal,
                X_test=X_test,
                y_test=y_test,
                target_name=target_name,
                horizon=horizon,
                issue_state_train=state_train,
                issue_state_val=state_val,
                issue_state_cal=state_cal,
                issue_state_test=state_test,
                ref_train=ref_train,
                ref_val=ref_val,
                ref_cal=ref_cal,
                ref_test=ref_test,
                random_state=RANDOM_STATE,
            )

            horizon_results[target_name] = {
                "result": result,
                "y_train": y_train,
                "y_val": y_val,
                "y_cal": y_cal,
                "y_test": y_test,
                "state_train": state_train,
                "state_val": state_val,
                "state_cal": state_cal,
                "state_test": state_test,
                "pred_dates_train": supervised.loc[idx_train, "pred_date"].reset_index(drop=True),
                "issue_dates_train": supervised.loc[idx_train, "issue_date"].reset_index(drop=True),
                "pred_dates_val": supervised.loc[idx_val, "pred_date"].reset_index(drop=True),
                "issue_dates_val": supervised.loc[idx_val, "issue_date"].reset_index(drop=True),
                "pred_dates_cal": supervised.loc[idx_cal, "pred_date"].reset_index(drop=True),
                "issue_dates_cal": supervised.loc[idx_cal, "issue_date"].reset_index(drop=True),
                "pred_dates_test": supervised.loc[idx_test, "pred_date"].reset_index(drop=True),
                "issue_dates_test": supervised.loc[idx_test, "issue_date"].reset_index(drop=True),
            }

            print(f"Selected candidate: {result['selected_candidate']}")
            print(f"Calibration info            : {result['calibration_info']}")

        if APPLY_QH_DIRECTION_CONSTRAINT and "Q_point" in horizon_results and "H_point" in horizon_results:
            q_result = horizon_results["Q_point"]["result"]
            h_result = horizon_results["H_point"]["result"]

            q_data = {
                "y_train": horizon_results["Q_point"]["y_train"],
                "y_val": horizon_results["Q_point"]["y_val"],
                "y_cal": horizon_results["Q_point"]["y_cal"],
                "y_test": horizon_results["Q_point"]["y_test"],
                "state_train": horizon_results["Q_point"]["state_train"],
                "state_val": horizon_results["Q_point"]["state_val"],
                "state_cal": horizon_results["Q_point"]["state_cal"],
                "state_test": horizon_results["Q_point"]["state_test"],
            }

            h_data = {
                "y_train": horizon_results["H_point"]["y_train"],
                "y_val": horizon_results["H_point"]["y_val"],
                "y_cal": horizon_results["H_point"]["y_cal"],
                "y_test": horizon_results["H_point"]["y_test"],
                "state_train": horizon_results["H_point"]["state_train"],
                "state_val": horizon_results["H_point"]["state_val"],
                "state_cal": horizon_results["H_point"]["state_cal"],
                "state_test": horizon_results["H_point"]["state_test"],
            }

            if ADAPTIVE_QH_DIRECTION_CONSTRAINT:
                apply_flag, qh_decision_info = should_apply_qh_direction_constraint(
                    horizon=horizon,
                    result_q=q_result,
                    result_h=h_result,
                    data_q=q_data,
                    data_h=h_data,
                )

                if apply_flag:
                    q_result, h_result, qh_df = apply_qh_direction_constraint_to_results(
                        horizon=horizon,
                        result_q=q_result,
                        result_h=h_result,
                        data_q=q_data,
                        data_h=h_data,
                    )

                    qh_df["adaptive_qh_constraint_applied"] = True
                    qh_df["before_joint_val_cal_score"] = qh_decision_info["before_joint_val_cal_score"]
                    qh_df["after_joint_val_cal_score"] = qh_decision_info["after_joint_val_cal_score"]
                    qh_df["relative_change"] = qh_decision_info["relative_change"]
                    qh_df["max_allowed_relative_degradation"] = qh_decision_info["max_allowed_relative_degradation"]

                    horizon_results["Q_point"]["result"] = q_result
                    horizon_results["H_point"]["result"] = h_result
                    all_qh_constraint.append(qh_df)

                else:
                    qh_df = pd.DataFrame(
                        [
                            {
                                "horizon": horizon,
                                "split": "all",
                                "adaptive_qh_constraint_applied": False,
                                "before_joint_val_cal_score": qh_decision_info["before_joint_val_cal_score"],
                                "after_joint_val_cal_score": qh_decision_info["after_joint_val_cal_score"],
                                "relative_change": qh_decision_info["relative_change"],
                                "max_allowed_relative_degradation": qh_decision_info["max_allowed_relative_degradation"],
                                "note": "Q-H direction constraint skipped because it exceeded allowed validation+calibration degradation.",
                            }
                        ]
                    )
                    all_qh_constraint.append(qh_df)

            else:
                q_result, h_result, qh_df = apply_qh_direction_constraint_to_results(
                    horizon=horizon,
                    result_q=q_result,
                    result_h=h_result,
                    data_q=q_data,
                    data_h=h_data,
                )
                qh_df["adaptive_qh_constraint_applied"] = True
                horizon_results["Q_point"]["result"] = q_result
                horizon_results["H_point"]["result"] = h_result
                all_qh_constraint.append(qh_df)

        for target_name in target_names:
            result = horizon_results[target_name]["result"]
            y_test = horizon_results[target_name]["y_test"]
            y_cal = horizon_results[target_name]["y_cal"]
            state_test = horizon_results[target_name]["state_test"]

            test_metrics = build_test_metrics(result, y_test, horizon, target_name)
            all_test_metrics.append(test_metrics)

            all_selected.append(build_selected_summary(horizon, target_name, result))

            if WRITE_INTERNAL_SELECTION_AUDIT:
                ranking = result["internal_candidate_ranking"].copy()
                ranking.insert(0, "horizon", horizon)
                ranking.insert(1, "target", target_name)
                all_candidate_rankings.append(ranking)

            cal_cand = result["calibration_candidate_metrics"].copy()
            cal_cand.insert(0, "horizon", horizon)
            cal_cand.insert(1, "target", target_name)
            cal_cand.insert(2, "model", PROPOSED_NAME)
            all_calibration_candidates.append(cal_cand)

            conformal_info = conformal_interval_summary(
                y_cal=y_cal,
                p_cal=result["y_pred_cal"],
                y_test=y_test,
                p_test=result["y_pred_test"],
                alpha=CONFORMAL_ALPHA,
                ref_cal=ref_cal,
                ref_test=ref_test,
            )

            all_conformal.append(
                pd.DataFrame(
                    [
                        {
                            "horizon": horizon,
                            "target": target_name,
                            "model": PROPOSED_NAME,
                            **conformal_info,
                        }
                    ]
                )
            )

            regime_df = build_regime_metrics(
                horizon=horizon,
                target_name=target_name,
                y_test=y_test,
                p_test=result["y_pred_test"],
                state_test=state_test,
                ref_cal=ref_cal,
                ref_test=ref_test,
            )

            if not regime_df.empty:
                all_regime_metrics.append(regime_df)

            for split_name in ["train", "val", "cal", "test"]:
                save_prediction_csv(
                    pred_dir=pred_dir,
                    horizon=horizon,
                    target_name=target_name,
                    split_name=split_name,
                    pred_dates=horizon_results[target_name][f"pred_dates_{split_name}"],
                    issue_dates=horizon_results[target_name][f"issue_dates_{split_name}"],
                    y_obs=horizon_results[target_name][f"y_{split_name}"],
                    result=result,
                )

            sw = result.get("stack_active_weights", [])
            if sw:
                sw_df = pd.DataFrame(sw)
                sw_df.insert(0, "horizon", horizon)
                sw_df.insert(1, "target", target_name)
                sw_df.insert(2, "selected_candidate", result["selected_candidate"])
                all_stack_weights.append(sw_df)

        if "Q_point" in horizon_results and "H_point" in horizon_results:
            pred_Q = horizon_results["Q_point"]["result"]["y_pred_test"]
            pred_H = horizon_results["H_point"]["result"]["y_pred_test"]

            audit_df = physical_consistency_audit(
                horizon=horizon,
                ref_test=ref_test,
                pred_Q_point=pred_Q,
                pred_H_point=pred_H,
            )
            all_physical_audit.append(audit_df)

    test_metrics_df = pd.concat(all_test_metrics, ignore_index=True) if all_test_metrics else pd.DataFrame()
    selected_df = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    candidate_ranking_df = pd.concat(all_candidate_rankings, ignore_index=True) if all_candidate_rankings else pd.DataFrame()
    calibration_candidates_df = pd.concat(all_calibration_candidates, ignore_index=True) if all_calibration_candidates else pd.DataFrame()
    conformal_df = pd.concat(all_conformal, ignore_index=True) if all_conformal else pd.DataFrame()
    regime_metrics_df = pd.concat(all_regime_metrics, ignore_index=True) if all_regime_metrics else pd.DataFrame()
    physical_audit_df = pd.concat(all_physical_audit, ignore_index=True) if all_physical_audit else pd.DataFrame()
    qh_constraint_df = pd.concat(all_qh_constraint, ignore_index=True) if all_qh_constraint else pd.DataFrame()
    stack_weights_df = pd.concat(all_stack_weights, ignore_index=True) if all_stack_weights else pd.DataFrame()
    split_diagnostics_df = pd.concat(all_split_diagnostics, ignore_index=True) if all_split_diagnostics else pd.DataFrame()
    split_df = pd.DataFrame(split_rows)

    runtime = time.time() - t0

    run_info_df = pd.DataFrame(
        [
            {
                "run_name": EXPERIMENT_TAG,
                "model_name": PROPOSED_NAME,
                "input_excel": str(INPUT_EXCEL),
                "output_dir": str(OUTPUT_DIR),
                "n_rows_daily": len(daily),
                "date_start": daily["Date"].min(),
                "date_end": daily["Date"].max(),
                "actual_horizons": json_dumps(HORIZONS),
                "target_policy": "h=1 uses Q_point and H_point; h=3 uses Q_point, H_point, Q_max, and H_max.",
                "n_features": len(feature_cols),
                "maximum_lag_or_window_days": max(max(LAGS), max(ROLL_WINDOWS)),
                "split_policy": "chronological_train_validation_calibration_test",
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "cal_ratio": CAL_RATIO,
                "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO - CAL_RATIO,
                "random_state": RANDOM_STATE,
                "model_reporting": "HydroPIML-RS predictions, diagnostics, and performance metrics.",
                "reporting_policy": "HydroPIML-RS predictions, diagnostics, and performance metrics are reported.",
                "internal_candidate_policy": (
                    "The proposed model internally uses raw learners, log-transform learners, "
                    "issue-state residual learners, inertia-shrinkage candidates, dry-low "
                    "conditional shrinkage candidates, metric-oriented stacks, and adaptive "
                    "leader blends. These are internal components, not external benchmark models."
                ),
                "add_inertia_shrinkage_candidates": ADD_INERTIA_SHRINKAGE_CANDIDATES,
                "add_dry_low_shrinkage_candidates": ADD_DRY_LOW_SHRINKAGE_CANDIDATES,
                "add_metric_oriented_stacks": ADD_METRIC_ORIENTED_STACKS,
                "add_adaptive_leader_blends": ADD_ADAPTIVE_LEADER_BLENDS,
                "allow_Q_affine_calibration": ALLOW_Q_AFFINE_CALIBRATION,
                "allow_H_affine_calibration": ALLOW_H_AFFINE_CALIBRATION,
                "allow_isotonic_calibration": ALLOW_ISOTONIC_CALIBRATION,
                "allow_quantile_mapping_calibration": ALLOW_QUANTILE_MAPPING_CALIBRATION,
                "calibration_selection_use_val_cal": CALIBRATION_SELECTION_USE_VAL_CAL,
                "calibration_selection_val_weight": CALIBRATION_SELECTION_VAL_WEIGHT,
                "calibration_selection_cal_weight": CALIBRATION_SELECTION_CAL_WEIGHT,
                "safe_calibration_identity_fallback": SAFE_CALIBRATION_ENABLE_IDENTITY_FALLBACK,
                "apply_qh_direction_constraint": APPLY_QH_DIRECTION_CONSTRAINT,
                "adaptive_qh_direction_constraint": ADAPTIVE_QH_DIRECTION_CONSTRAINT,
                "qh_constraint_max_relative_degradation": QH_CONSTRAINT_MAX_RELATIVE_DEGRADATION,
                "conformal_alpha": CONFORMAL_ALPHA,
                "test_policy": "The independent test period is used only for final evaluation.",
                "optional_internal_learner_status": json_dumps(OPTIONAL_MODEL_STATUS),
                "runtime_sec": runtime,
            }
        ]
    )

    feature_info_df = pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_cols)),
            "feature_name": feature_cols,
        }
    )

    summary_path = out_dir / f"{EXPERIMENT_TAG}_summary.xlsx"

    with pd.ExcelWriter(summary_path) as writer:
        run_info_df.to_excel(writer, sheet_name=safe_sheet_name("run_info"), index=False)
        feature_info_df.to_excel(writer, sheet_name=safe_sheet_name("feature_info"), index=False)
        split_df.to_excel(writer, sheet_name=safe_sheet_name("splits"), index=False)

        if not split_diagnostics_df.empty:
            split_diagnostics_df.to_excel(writer, sheet_name=safe_sheet_name("split_distribution"), index=False)

        test_metrics_df.to_excel(writer, sheet_name=safe_sheet_name("test_metrics"), index=False)
        selected_df.to_excel(writer, sheet_name=safe_sheet_name("selected_candidate"), index=False)

        if WRITE_INTERNAL_SELECTION_AUDIT and not candidate_ranking_df.empty:
            candidate_ranking_df.to_excel(writer, sheet_name=safe_sheet_name("candidate_selection_audit"), index=False)

        if not calibration_candidates_df.empty:
            calibration_candidates_df.to_excel(writer, sheet_name=safe_sheet_name("calibration_audit"), index=False)

        if not conformal_df.empty:
            conformal_df.to_excel(writer, sheet_name=safe_sheet_name("conformal_intervals"), index=False)

        if not regime_metrics_df.empty:
            regime_metrics_df.to_excel(writer, sheet_name=safe_sheet_name("regime_metrics"), index=False)

        if not physical_audit_df.empty:
            physical_audit_df.to_excel(writer, sheet_name=safe_sheet_name("physical_audit"), index=False)

        if not qh_constraint_df.empty:
            qh_constraint_df.to_excel(writer, sheet_name=safe_sheet_name("qh_direction_constraint"), index=False)

        if not stack_weights_df.empty:
            stack_weights_df.to_excel(writer, sheet_name=safe_sheet_name("stack_weights"), index=False)

    print_header("HydroPIML-RS prediction experiment completed")
    print(f"Summary Excel : {summary_path}")
    print(f"Prediction dir: {pred_dir}")
    print(f"Reference dir : {ref_dir}")
    print(f"Runtime       : {runtime:.2f} seconds")

    print_section("Chronological split")
    print(split_df.to_string(index=False))

    print_section("HydroPIML-RS independent test metrics")
    if not test_metrics_df.empty:
        print(test_metrics_df.to_string(index=False))

    print_section("Selected candidates")
    if not selected_df.empty:
        print(selected_df.to_string(index=False))

    print_header("Reproducibility note")
    print(
        "HydroPIML-RS uses a leakage-free chronological protocol. "
        "The independent test period was not used for model fitting, internal candidate selection, "
        "calibration  fitting, calibration-method selection, ensemble optimization, threshold optimization, "
        "or Q-H physical-consistency decision."
    )


if __name__ == "__main__":
    main()