import joblib
import numpy as np
import pandas as pd
import os
import re
from app.features.recommendation.common.session import generate_session_id
from app.features.recommendation.llm.explainers import (
    explain_prediction_results
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models")

def map_age_to_trained_group(age_input):
    """محرك ذكي لتحويل أي رقم أو نص للفئة العمرية المناسبة"""
    age_str = str(age_input).strip().lower()
    numbers = re.findall(r'\d+', age_str)
    if not numbers:
        return "25-34"
    
    age = int(numbers[0])
    if 18 <= age <= 24: return "18-24"
    if 25 <= age <= 34: return "25-34"
    if 35 <= age <= 44: return "35-44"
    return "45-60"

def predict_campaign_kpis(user_input: dict,explain_prediction=False):
    session_id = generate_session_id()
    model = joblib.load(os.path.join(MODEL_PATH, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_PATH, "label_encoders.pkl"))

    platform_mapper = {
        'tiktok ads': 'instagram', 'snapchat ads': 'instagram', 'influencer marketing': 'instagram',
        'linkedin ads': 'facebook', 'youtube ads': 'facebook', 'google ads': 'facebook',
        'amazon ads': 'pinterest', 'jumia ads': 'pinterest', 'noon ads': 'pinterest',
        'native ads': 'twitter', 'twitter (x) ads': 'twitter'
    }
    
    original_platform = str(user_input.get('Platform_Name', 'facebook')).lower().strip()
    trained_platforms = encoders['Platform_Name'].classes_
    
    if original_platform in trained_platforms:
        platform_to_use = original_platform
    else:
        platform_to_use = platform_mapper.get(original_platform, 'facebook')
    
    user_input['Platform_Name'] = platform_to_use

    user_input['Age_Group'] = map_age_to_trained_group(user_input.get('Age_Group'))

    data = pd.DataFrame([user_input])
    data['Daily_Budget'] = data['Total_Budget'] / data['duration_days']

    columns_to_encode = ['Platform_Name', 'interest', 'Gender', 'Objective', 'Age_Group']
    for col in columns_to_encode:
        val = str(data.at[0, col]).lower().strip()
        if val in encoders[col].classes_:
            data[col] = encoders[col].transform([val])[0]
        else:
            data[col] = encoders[col].transform([encoders[col].classes_[0]])[0]
    
    data[['Total_Budget', 'Daily_Budget']] = scaler.transform(data[['Total_Budget', 'Daily_Budget']])
    
    feature_order = [
        'Platform_Name', 'interest', 'Total_Budget', 'Age_Group', 
        'duration_days', 'Gender', 'Objective', 'Daily_Budget'
    ]
    
    prediction = model.predict(data[feature_order])
    
    impr = np.expm1(prediction[0][0])
    clicks = np.expm1(prediction[0][1])
    convs = np.expm1(prediction[0][2])
    rev = np.expm1(prediction[0][3])
    spent = np.expm1(prediction[0][4])

    impr, clicks, convs, spent = [max(x, 1.0) for x in [impr, clicks, convs, spent]]

    results = []
    results.append({
        "Expected_Impressions": int(round(impr, 0)),
        "Expected_Clicks": int(round(clicks, 0)),
        "Expected_Conversions": int(round(convs, 0)),
        "Expected_Spent": float(round(spent, 2)),
        "Expected_Revenue": float(round(rev, 2)),
        "Conversion_Rate": int(round((convs / clicks) * 100, 0)),
        "ROI": float(round((rev - spent) / spent, 4)),
        "CTR": float(round((clicks / impr) * 100, 2)),
        "CPC": float(round(spent / clicks, 2)),
        "CPA": float(round(spent / convs, 2))
    })

    explanation = None
    if explain_prediction:
        explanation = explain_prediction_results(user_input, results[0])


    return {
        "session_id": session_id,
        "prediction": results[0],
        "explanation": explanation

    }