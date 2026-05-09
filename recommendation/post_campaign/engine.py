import pandas as pd

from recommendation.common.data_prep import prepare_data
from recommendation.common.normalization import normalize
from recommendation.common.scoring import compute_success_score
from recommendation.post_campaign.pca_weights import compute_pca_weights
from recommendation.post_campaign.classifier import classify
from recommendation.llm.explainers import explain_failed_summary_from_stats
from recommendation.common.session import generate_session_id
from recommendation.post_campaign.driver_analysis import compute_failure_drivers
from recommendation.llm.explainers import explain_failure_drivers


SUCCESS_FEATURES = [
    "ROI",
    "Conversion_Rate",
    "CTR",
    "CPA"
]


def run_post_campaign(df, explain_failed_summary=False):
    session_id = generate_session_id()

    clean = prepare_data(df)
    clean = clean.dropna(subset=SUCCESS_FEATURES)
    
    meta_cols = ["campaign_ID", "Campaign_Name"]
    meta_df = clean[meta_cols].copy()

    norm = normalize(clean)
    norm_SUCCESS_FEATURES = norm[SUCCESS_FEATURES].copy()
    print(norm_SUCCESS_FEATURES)
    weights = compute_pca_weights(norm_SUCCESS_FEATURES)
    norm_SUCCESS_FEATURES["Success_Score"] = compute_success_score(norm_SUCCESS_FEATURES, weights)
    norm_SUCCESS_FEATURES["Status"] = classify(norm_SUCCESS_FEATURES["Success_Score"])
    print(norm_SUCCESS_FEATURES)
    final_df = meta_df.reset_index(drop=True).join(
    norm_SUCCESS_FEATURES[["Success_Score", "Status"]].reset_index(drop=True)
    )
    
    #failed_summary = None
    driver_summary = None

    if explain_failed_summary:
        #old
        failed_df = norm_SUCCESS_FEATURES[norm_SUCCESS_FEATURES["Status"] == "failed"]
        print(failed_df)
        #stats = compute_failed_stats(failed_df)
        #failed_summary = explain_failed_summary_from_stats(stats)
        #new
        driver_stats = compute_failure_drivers(clean, norm_SUCCESS_FEATURES["Status"])
        driver_summary = explain_failure_drivers(driver_stats)

    return {
        "session_id": session_id,
        "results": final_df,
        "weights": weights,
        #"failed_summary": failed_summary,
        "driver_summary": driver_summary
    }


#def compute_failed_stats(failed_df):
    return {
        "count": len(failed_df),
        "avg_roi": float(failed_df["ROI"].mean()),
        "avg_ctr": float(failed_df["CTR"].mean()),
        "avg_conversion_rate": float(failed_df["Conversion_Rate"].mean()),
        "avg_cpa": float(failed_df["CPA"].mean()),

        "low_roi_ratio": float((failed_df["ROI"] < 0.3).mean()),
        "low_ctr_ratio": float((failed_df["CTR"] < 0.3).mean()),
        "low_conv_ratio": float((failed_df["Conversion_Rate"] < 0.05).mean()),
        "high_cpa_ratio": float((failed_df["CPA"] < 0.3).mean()),
    }
