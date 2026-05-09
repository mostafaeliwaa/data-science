import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric columns of `df` using MinMaxScaler.

    Non-numeric columns are ignored so string identifiers like campaign IDs
    are not passed to scikit-learn scalers.
    """
    scaler = MinMaxScaler()

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns available for normalization")

    scaled = scaler.fit_transform(df[numeric_cols])
    norm = pd.DataFrame(scaled, columns=numeric_cols, index=df.index)

    if "CPA" in norm.columns:
        norm["CPA"] = 1 - norm["CPA"]

    return norm

INFLIGHT_KPI_COLS = [
    "roi",
    "ctr",
    "conversion_rate"
]

def normalize_inflight(df: pd.DataFrame):
    df = df.copy()

    scaler = MinMaxScaler()

    df_norm = scaler.fit_transform(df[INFLIGHT_KPI_COLS])

    df["roi_norm"] = df_norm[:, 0]
    df["ctr_norm"] = df_norm[:, 1]
    df["conversion_norm"] = df_norm[:, 2]

    return df
