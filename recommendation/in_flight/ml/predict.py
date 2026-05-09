import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_PATH = Path(__file__).parent
roi_model = joblib.load(BASE_PATH / "roi_model.pkl")
conv_model = joblib.load(BASE_PATH / "conversion_model.pkl")
scaler = joblib.load(BASE_PATH / "scaler.pkl")


FEATURES = [
    "time_elapsed_ratio",
    "budget_spent_ratio",
    "impressions_so_far",
    "clicks_so_far",
    "revenue_so_far",
    "ctr",
    "cpc",
    "conversion_rate"
]


def predict_inflight_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = df[FEATURES].copy()
    X_scaled = scaler.transform(X)
    df["expected_final_roi"] = roi_model.predict(X_scaled)
    df["expected_final_conversions_log"] = conv_model.predict(X_scaled)
    df["expected_final_conversions"] = np.expm1(
        df["expected_final_conversions_log"]
    )

    return df
