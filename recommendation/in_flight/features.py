import pandas as pd
from datetime import datetime
import numpy as np

def enrich_inflight_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Start_Date"] = pd.to_datetime(df["Start_Date"])
    df["End_Date"] = pd.to_datetime(df["End_Date"])
    df["current_date"] = pd.to_datetime(datetime.today().date())
    #df["current_date"] = pd.to_datetime(df["current_date"])

    total_days = (df["End_Date"] - df["Start_Date"]).dt.days
    elapsed_days = (df["current_date"] - df["Start_Date"]).dt.days
    # avoid division by zero
    total_days = total_days.replace(0, 1)
    # time ratio (0 → 1)
    df["time_elapsed_ratio"] = elapsed_days / total_days
    df["time_elapsed_ratio"] = df["time_elapsed_ratio"].clip(0, 1)

    df["budget_spent_ratio"] = df["Budget_Spent"] / df["Total_Budget"]
    df["impressions_so_far"] = df["Impressions"]
    df["clicks_so_far"] = df["Clicks"]
    df["revenue_so_far"] = df["Revenue"]
    df["ctr"] = df["Clicks"] / df["Impressions"]
    df["conversion_rate"] = df["Conversions"] / df["Clicks"]
    df["cpc"] = df["Budget_Spent"] / df["Clicks"]
    df["roi"] = df["Revenue"] / df["Budget_Spent"]

    # clean infinities & NaNs
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    return df

def build_inflight_training_features(df):
    rows = []

    SNAPSHOTS = [0.25, 0.5, 0.75]

    for _, r in df.iterrows():
        for t in SNAPSHOTS:
            rows.append({
                
                "time_elapsed_ratio": t,
                "budget_spent_ratio": t * (r["Budget_Spent"] / r["Total_Budget"]),
                "impressions_so_far": t * r["Impressions"],
                "clicks_so_far": t * r["Clicks"],
                "revenue_so_far": t * r["Revenue"],
                "ctr": r["CTR"],
                "cpc": r["CPC"],
                "conversion_rate": r["Conversion_Rate"],
                "target_roi": r["ROI"],
                "target_conversions": r["Conversions"]
            })

    new_df = pd.DataFrame(rows)
    new_df["target_conversions"] = np.log1p(new_df["target_conversions"])

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

    X = new_df[FEATURES]
    y_roi = new_df["target_roi"]
    y_conv = new_df["target_conversions"]

    return X, y_roi, y_conv