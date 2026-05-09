import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from dotenv import load_dotenv

# إضافة مسار جذر المشروع إلى sys.path حتى يتعرف بايثون على مجلد "app" الرئيسي
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# تحميل ملف الـ .env لقراءة مفاتيح الـ API الخاصة بالـ LLM
load_dotenv(os.path.join(root_dir, ".env"))
load_dotenv()

from typing import Literal, Union
import tempfile

# Import the same functions used in main.py
from recommendation.pre_campaign.engine import predict_campaign_kpis

# Set page configuration
st.set_page_config(
    page_title="Marketing Recommendation System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2196F3;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FFC107;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 Marketing Recommendation System")
st.sidebar.markdown("---")

st.sidebar.info(
    """
    This application provides AI-powered marketing campaign analysis:
    - **Pre-Campaign**: Predict KPIs for planned campaigns before execution
    """
)

# Main title
st.markdown('<h1 class="main-header">📊 Marketing Recommendation System</h1>', unsafe_allow_html=True)

# Pre-Campaign Prediction Section
st.markdown('<h2 class="section-header">Pre-Campaign KPI Prediction</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<b>Fill in campaign details</b> to predict key performance indicators (KPIs) 
before launching your campaign. The model will estimate ROI, conversion rate, 
CTR, and CPA based on historical data.
</div>
""", unsafe_allow_html=True)

# Create form for pre-campaign prediction
with st.form("pre_campaign_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        platform_name = st.selectbox(
            "Platform Name",
            ["facebook", "instagram", "pinterest", "twitter", 
             "google ads", "youtube ads", "tiktok ads", "snapchat ads", 
             "linkedin ads", "amazon ads", "jumia ads", "noon ads", 
             "native ads", "influencer marketing", "other"]
        )
        
        interest = st.selectbox(
            "Interest",
            ["health", "home", "food", "fashion", "technology"]
        )
        
        objective = st.selectbox(
            "Objective",
            ["brand awareness", "product launch", "increase sales", "market expansion"]
        )
    
    with col2:
        gender = st.selectbox(
            "Gender",
            ["female", "male", "all"]
        )
        
        age_group = st.text_input(
            "Age Group",
            value="25-34",
            help="Enter age group as string (e.g., '25-34') or as integer"
        )
        
        total_budget = st.number_input(
            "Total Budget ($)",
            min_value=100.0,
            max_value=1000000.0,
            value=10000.0,
            step=1000.0
        )
        
        duration_days = st.number_input(
            "Duration (days)",
            min_value=1,
            max_value=365,
            value=30,
            step=1
        )
    
    explain_prediction = st.checkbox("Explain prediction", value=False)
    
    submitted = st.form_submit_button("🎯 Predict KPIs", type="primary")

if submitted:
    with st.spinner("Predicting campaign KPIs..."):
        try:
            # Prepare input data
            user_input = {
                "Platform_Name": platform_name,
                "interest": interest,
                "Objective": objective,
                "Gender": gender,
                "Age_Group": age_group,
                "Total_Budget": float(total_budget),
                "duration_days": int(duration_days)
            }
            
            # Call prediction function
            output = predict_campaign_kpis(
                user_input,
                explain_prediction=explain_prediction
            )
            
            # Display results
            st.markdown('<h3 class="section-header">📊 Prediction Results</h3>', unsafe_allow_html=True)
            
            # Prediction metrics
            st.markdown("### Predicted KPIs")
            prediction = output['prediction']
            
            if isinstance(prediction, dict):
                # Create metrics display
                col1, col2, col3, col4 = st.columns(4)
                
                metrics = {
                    "ROI": prediction.get('ROI', 0),
                    "Conversion Rate": prediction.get('Conversion_Rate', 0),
                    "CTR": prediction.get('CTR', 0),
                    "CPA": prediction.get('CPA', 0)
                }
                
                for (metric_name, value), col in zip(metrics.items(), [col1, col2, col3, col4]):
                    with col:
                        if metric_name == "ROI":
                            st.metric("ROI", f"{value:.2f}x", delta=None)
                        elif metric_name == "Conversion Rate":
                            st.metric("Conversion Rate", f"{value:.2%}")
                        elif metric_name == "CTR":
                            st.metric("CTR", f"{value:.2%}")
                        elif metric_name == "CPA":
                            st.metric("CPA", f"${value:.2f}")
                
                # Show all prediction data
                with st.expander("View full prediction data"):
                    st.json(prediction)
            else:
                st.write(prediction)
            
            # Explanation if requested
            if explain_prediction and output.get('explanation'):
                st.markdown("### 🤖 Prediction Explanation")
                st.info(output['explanation'])
            
            # Campaign success assessment
            st.markdown("### 🎯 Campaign Success Assessment")
            
            # Simple heuristic based on ROI
            if isinstance(prediction, dict) and 'ROI' in prediction:
                roi = prediction['ROI']
                if roi > 3.0:
                    st.success("🎉 **Excellent ROI Expected**: This campaign has high potential for success!")
                elif roi > 1.5:
                    st.info("📈 **Good ROI Expected**: This campaign shows promising returns.")
                elif roi > 1.0:
                    st.warning("⚠️ **Moderate ROI Expected**: Consider optimizing your campaign strategy.")
                else:
                    st.error("🔴 **Low ROI Expected**: Review your campaign parameters before proceeding.")
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            st.markdown("""
            - **Consider A/B testing** different creatives and targeting options
            - **Monitor performance weekly** and adjust budget allocation
            - **Use retargeting** for users who show interest but don't convert
            - **Optimize landing pages** to improve conversion rates
            """)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Marketing Recommendation System • Built with Streamlit • Powered by AI/ML</p>
</div>
""", unsafe_allow_html=True)