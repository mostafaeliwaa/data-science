import pandas as pd

NUM_DRIVER_COLS = [
    "Total_Budget",
    "Budget_Spent",
    "CPC",
    "duration_days",
    "engagement_score"
]

CAT_DRIVER_COLS = [
    "Platform_Name",
    "Objective",
    "Age_Group",
    "Gender",
    "Location",
    "Language"
]


def compute_failure_drivers(clean_df: pd.DataFrame, status_series: pd.Series) -> dict:
    failed = clean_df[status_series == "failed"]
    success = clean_df[status_series == "successful"]

    insights = {}

    # Numerical comparison
    for col in NUM_DRIVER_COLS:
        if col in clean_df.columns:
            insights[f"{col}_failed_mean"] = float(failed[col].mean())
            insights[f"{col}_success_mean"] = float(success[col].mean())

    # Categorical dominant patterns
    for col in CAT_DRIVER_COLS:
        if col in clean_df.columns:
            insights[f"{col}_failed_top"] = (
                failed[col].mode().iloc[0] if not failed[col].mode().empty else None
            )
            insights[f"{col}_success_top"] = (
                success[col].mode().iloc[0] if not success[col].mode().empty else None
            )

    return insights
