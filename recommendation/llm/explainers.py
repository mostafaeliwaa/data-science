from recommendation.llm.client import call_llm
from recommendation.llm.prompts import failed_summary_prompt, failure_driver_prompt, inflight_summary_prompt, pre_campaign_prediction_prompt

def explain_failed_summary_from_stats(stats: dict) -> str:
    if stats["count"] == 0:
        return "No failed campaigns detected."

    prompt = failed_summary_prompt(stats)
    return call_llm(prompt)

def explain_failure_drivers(driver_stats: dict) -> str:
    prompt = failure_driver_prompt(driver_stats)
    return call_llm(prompt)

#senario2
def explain_inflight_summary_from_stats(stats: dict) -> str:
    """
    stats example:
    {
        "total_campaigns": 10,
        "keep": 4,
        "optimize": 3,
        "stop": 3,
        "avg_score": 0.46
    }
    """

    if stats["total_campaigns"] == 0:
        return "No active campaigns were provided for analysis."

    prompt = inflight_summary_prompt(stats)
    return call_llm(prompt)


#senario3
def explain_prediction_results(inputs: dict, results: dict) -> str:

    prompt = pre_campaign_prediction_prompt(inputs, results)
    return call_llm(prompt)