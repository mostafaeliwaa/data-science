import pandas as pd
from recommendation.common.session import generate_session_id
from recommendation.in_flight.features import enrich_inflight_features
from recommendation.common.normalization import normalize_inflight
from recommendation.common.scoring import (
    performance_score, pace_score, potential_score
)
from recommendation.in_flight.decision import make_decision
from recommendation.in_flight.pca_weights import learn_inflight_pca_weights
from recommendation.llm.explainers import (
    explain_inflight_summary_from_stats
)
from recommendation.in_flight.ml.predict import predict_inflight_kpis


def run_inflight_from_csv(df: pd.DataFrame,explain_inflight_summary=False):
    session_id = generate_session_id()

    df = enrich_inflight_features(df)
    df = normalize_inflight(df)
    df = predict_inflight_kpis(df)

    df["performance_score"] = df.apply(
        lambda r: performance_score(r["roi_norm"], r["ctr_norm"], r["conversion_norm"]),
        axis=1
    )

    df["pace_score"] = df.apply(
        lambda r: pace_score(r["budget_spent_ratio"], r["time_elapsed_ratio"]),
        axis=1
    )

    df["potential_score"] = df["time_elapsed_ratio"].apply(potential_score)
    
    score_df = df[["performance_score", "pace_score", "potential_score"]]
    weights = learn_inflight_pca_weights(score_df)

   
    results = []

    for _, row in df.iterrows():
        final_score = (
            row["performance_score"] * weights["performance"]
            + row["pace_score"] * weights["pace"]
            + row["potential_score"] * weights["potential"]
        )

        decision = make_decision(final_score)

        results.append({
            "campaign_id": row["campaign_ID"],
            "Campaign_Name": row["Campaign_Name"],
            "final_score": round(final_score, 3),
            "decision": decision,
            "expected_final_roi": round(row["expected_final_roi"], 3)
        })
    
    summary = build_inflight_summary(results)
    inflight_summary=None

    if explain_inflight_summary:
        inflight_summary = explain_inflight_summary_from_stats(summary)


    return {
        "session_id": session_id,
        "weights": weights,
        "results": results,
        "summary": summary,
        "inflight_summary": inflight_summary
    }

def build_inflight_summary(results: list):
    total = len(results)

    keep = sum(1 for r in results if r["decision"] == "KEEP")
    optimize = sum(1 for r in results if r["decision"] == "OPTIMIZE")
    stop = sum(1 for r in results if r["decision"] == "STOP")

    avg_score = sum(r["final_score"] for r in results) / total if total else 0
    predict_roi = [r["expected_final_roi"] for r in results if r["expected_final_roi"] is not None]
    avg_roi = sum(predict_roi) / len(predict_roi) if predict_roi else 0

    return {
        "total_campaigns": total,
        "keep": keep,
        "optimize": optimize,
        "stop": stop,
        "avg_score": round(avg_score, 3),
        "avg_expected_final_roi": round(avg_roi, 3)
    }
