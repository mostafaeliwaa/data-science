import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from recommendation.pre_campaign.engine import predict_campaign_kpis
from recommendation.in_flight.engine import run_inflight_from_csv
from recommendation.post_campaign.engine import run_post_campaign

load_dotenv()


def _pre_campaign_tab():
    st.markdown("### 🎯 Pre-Campaign KPI Prediction")
    st.caption("Estimate ROI, conversion rate, CTR, and CPA before launching.")

    with st.form("pre_campaign_form"):
        col1, col2 = st.columns(2)
        with col1:
            platform_name = st.selectbox(
                "Platform",
                [
                    "facebook", "instagram", "pinterest", "twitter",
                    "google ads", "youtube ads", "tiktok ads", "snapchat ads",
                    "linkedin ads", "amazon ads", "jumia ads", "noon ads",
                    "native ads", "influencer marketing", "other",
                ],
            )
            interest = st.selectbox("Interest", ["health", "home", "food", "fashion", "technology"])
            objective = st.selectbox(
                "Objective",
                ["brand awareness", "product launch", "increase sales", "market expansion"],
            )
        with col2:
            gender = st.selectbox("Gender", ["female", "male", "all"])
            age_group = st.text_input("Age group", value="25-34")
            total_budget = st.number_input("Total budget ($)", 100.0, 1_000_000.0, 10_000.0, 1_000.0)
            duration_days = st.number_input("Duration (days)", 1, 365, 30, 1)

        explain = st.checkbox("Explain prediction (LLM)", value=False)
        submitted = st.form_submit_button("Predict KPIs", type="primary")

    if not submitted:
        return

    user_input = {
        "Platform_Name": platform_name,
        "interest": interest,
        "Objective": objective,
        "Gender": gender,
        "Age_Group": age_group,
        "Total_Budget": float(total_budget),
        "duration_days": int(duration_days),
    }

    with st.spinner("Predicting…"):
        try:
            output = predict_campaign_kpis(user_input, explain_prediction=explain)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)
            return

    pred = output["prediction"]
    if isinstance(pred, dict):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROI", f"{pred.get('ROI', 0):.2f}x")
        c2.metric("Conversion rate", f"{pred.get('Conversion_Rate', 0)}%")
        c3.metric("CTR", f"{pred.get('CTR', 0)}%")
        c4.metric("CPA", f"${pred.get('CPA', 0):.2f}")
        with st.expander("Full prediction"):
            st.json(pred)

        roi = pred.get("ROI", 0)
        if roi > 3:
            st.success("Excellent ROI expected — strong launch signal.")
        elif roi > 1.5:
            st.info("Good ROI expected — proceed with monitoring.")
        elif roi > 1:
            st.warning("Moderate ROI — consider optimizing targeting.")
        else:
            st.error("Low ROI — review parameters before launching.")
    else:
        st.write(pred)

    if explain and output.get("explanation"):
        st.markdown("#### 🤖 LLM Explanation")
        st.info(output["explanation"])


def _in_flight_tab():
    st.markdown("### 🛫 In-Flight Campaign Optimization")
    st.caption("Upload an active-campaign CSV to get KEEP / OPTIMIZE / STOP recommendations.")

    uploaded = st.file_uploader("Active campaigns CSV", type=["csv"], key="inflight_csv")
    explain = st.checkbox("Generate LLM summary", value=False, key="inflight_explain")

    if uploaded is None:
        return
    if st.button("Analyze in-flight campaigns", type="primary"):
        df = pd.read_csv(io.BytesIO(uploaded.read()))
        with st.spinner("Scoring…"):
            try:
                output = run_inflight_from_csv(df, explain_inflight_summary=explain)
            except Exception as e:
                st.error(f"In-flight analysis failed: {e}")
                st.exception(e)
                return

        summary = output["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", summary["total_campaigns"])
        c2.metric("Keep", summary["keep"])
        c3.metric("Optimize", summary["optimize"])
        c4.metric("Stop", summary["stop"])

        st.markdown("#### Per-campaign decisions")
        st.dataframe(pd.DataFrame(output["results"]), use_container_width=True)

        with st.expander("Score weights (PCA)"):
            st.json(output["weights"])

        if explain and output.get("inflight_summary"):
            st.markdown("#### 🤖 LLM Summary")
            st.info(output["inflight_summary"])


def _post_campaign_tab():
    st.markdown("### 📊 Post-Campaign Audit")
    st.caption("Upload a finished-campaign CSV to classify success and surface failure drivers.")

    uploaded = st.file_uploader("Finished campaigns CSV", type=["csv"], key="post_csv")
    explain = st.checkbox("Explain failure drivers (LLM)", value=False, key="post_explain")

    if uploaded is None:
        return
    if st.button("Run post-campaign audit", type="primary"):
        df = pd.read_csv(io.BytesIO(uploaded.read()))
        with st.spinner("Analyzing…"):
            try:
                output = run_post_campaign(df, explain_failed_summary=explain)
            except Exception as e:
                st.error(f"Post-campaign analysis failed: {e}")
                st.exception(e)
                return

        st.markdown("#### Classified campaigns")
        st.dataframe(output["results"], use_container_width=True)

        with st.expander("Feature weights (PCA)"):
            st.json(output["weights"])

        if explain and output.get("driver_summary"):
            st.markdown("#### 🤖 Failure-Driver Analysis")
            st.info(output["driver_summary"])


def render():
    st.markdown("## 📈 Recommendation Models")
    tab1, tab2, tab3 = st.tabs(["Pre-Campaign", "In-Flight", "Post-Campaign"])
    with tab1:
        _pre_campaign_tab()
    with tab2:
        _in_flight_tab()
    with tab3:
        _post_campaign_tab()
