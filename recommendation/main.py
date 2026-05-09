from fastapi import FastAPI, UploadFile, File, Query
from pydantic import BaseModel
import pandas as pd
from typing import Literal, Union
from app.features.recommendation.post_campaign.engine import run_post_campaign
from app.features.recommendation.llm.client import call_llm
from app.features.recommendation.in_flight.engine import run_inflight_from_csv
from app.features.recommendation.pre_campaign.engine import predict_campaign_kpis
app = FastAPI()


@app.post("/recommendation/post-campaign/upload")
async def post_campaign_from_csv(
    file: UploadFile = File(...),
    explain_failed_summary: bool = Query(False)
):
    df = pd.read_csv(file.file)

    output = run_post_campaign(
        df,
        explain_failed_summary=explain_failed_summary
    )

    response = {
        "session_id": output["session_id"],
        "weights": output["weights"],
        "results": output["results"].to_dict(orient="records")
    }

    if explain_failed_summary:
       # response["failed_summary"] = output["failed_summary"]
        response["driver_summary"] = output["driver_summary"]

    return response


@app.post("/recommendation/in-flight/upload")
async def inflight_from_csv(file: UploadFile = File(...),
                            explain_inflight_summary: bool = Query(False)):
    df = pd.read_csv(file.file)

    output = run_inflight_from_csv(
        df,
        explain_inflight_summary=explain_inflight_summary
    )

    response = {
        "session_id": output["session_id"],
        "weights": output["weights"],
        "results": output["results"],
        "summary": output["summary"]
    }

    if explain_inflight_summary:
        response["inflight_summary"] = output["inflight_summary"]


    return response

class PreCampaignInput(BaseModel):
    Platform_Name: Literal[
        "facebook", "instagram", "pinterest", "twitter", 
        "google ads", "youtube ads", "tiktok ads", "snapchat ads", 
        "linkedin ads", "amazon ads", "jumia ads", "noon ads", 
        "native ads", "influencer marketing", "other"
    ]
    interest: Literal["health", "home", "food", "fashion", "technology"]
    Objective: Literal["brand awareness", "product launch", "increase sales", "market expansion"]
    Gender: Literal["female", "male", "all"]
    Age_Group: Union[str, int]
    Total_Budget: float
    duration_days: int

@app.post("/recommendation/pre-campaign/predict")
async def pre_campaign_predict(input_data: PreCampaignInput, explain_prediction: bool = Query(False)):
    user_input = input_data.dict()
    output = predict_campaign_kpis(
        user_input,
        explain_prediction=explain_prediction
    )

    response = {
        "session_id": output["session_id"],
        "prediction": output["prediction"]
    }

    if explain_prediction:
        response["explanation"] = output["explanation"]

    return response 

@app.get("/llm-test")
def llm_test():
    return {
        "response": call_llm("Say hello in one sentence.")
    }

