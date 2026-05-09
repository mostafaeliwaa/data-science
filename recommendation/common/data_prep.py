import pandas as pd

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    # normalize column names (strip whitespace)
    data.columns = [c.strip() for c in data.columns]

    # parse date columns
    for dt_col in ("Start_Date", "End_Date"):
        if dt_col in data.columns:
            data[dt_col] = pd.to_datetime(data[dt_col], errors="coerce")

    # compute duration_days if not present
    if "duration_days" not in data.columns and {
        "Start_Date",
        "End_Date",
    }.issubset(set(data.columns)):
        data["duration_days"] = (data["End_Date"] - data["Start_Date"]).dt.days

    # convert numeric-like columns
    num_cols = [
        "Total_Budget",
        "Budget_Spent",
        "Expected_Budget",
        "Impressions",
        "Clicks",
        "Results",
        "Revenue",
        "CTR",
        "CPC",
        "CPR",
        "Conversion_Rate",
        "engagement_score",
        "ROI",
        "duration_days",
    ]
    for c in num_cols:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    # remove exact duplicates (prefer first occurrence)
    if "campaign_ID" in data.columns:
        subset = [c for c in ("campaign_ID", "Start_Date", "End_Date") if c in data.columns]
        data = data.drop_duplicates(subset=subset)
    else:
        data = data.drop_duplicates()

    # filter out rows with invalid KPI values
    if "ROI" in data.columns and "Conversion_Rate" in data.columns:
        data = data.dropna(subset=["ROI", "Conversion_Rate"])
        data = data[(data["ROI"] >= 0) & (data["Conversion_Rate"] >= 0)]

    # drop rows missing essential counts
    essential = [c for c in ("Impressions", "Clicks") if c in data.columns]
    if essential:
        data = data.dropna(subset=essential)

    data = data.reset_index(drop=True)
    return data
