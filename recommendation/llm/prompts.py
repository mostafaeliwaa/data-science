def failed_summary_prompt(stats: dict) -> str:
    return f"""
You are a senior digital marketing analyst.

We analyzed FAILED marketing campaigns with the following aggregated insights:

- Total failed campaigns: {stats['count']}
- Average ROI: {stats['avg_roi']:.2f}
- Average CTR: {stats['avg_ctr']:.2f}
- Average Conversion Rate: {stats['avg_conversion_rate']:.2f}
- Average CPA Efficiency: {stats['avg_cpa']:.2f}

Failure patterns (percentage of failed campaigns):
- Low ROI campaigns: {stats['low_roi_ratio']*100:.1f}%
- Low CTR campaigns: {stats['low_ctr_ratio']*100:.1f}%
- Low Conversion Rate campaigns: {stats['low_conv_ratio']*100:.1f}%
- High CPA inefficiency: {stats['high_cpa_ratio']*100:.1f}%

Tasks:
1. Identify the main reasons behind campaign failures
2. Explain the dominant failure patterns
3. Provide clear, actionable recommendations to improve future campaigns

Do NOT mention numbers explicitly in the answer.
Focus on business insights and solutions.
"""

def failure_driver_prompt(driver_stats: dict) -> str:
    return f"""
You are a senior marketing performance strategist.

We analyzed failed campaigns and compared them with successful ones.
Your goal is to identify ROOT CAUSES behind campaign failure.

Differences observed:

Budget & Efficiency:
- Failed campaigns budget: {driver_stats.get("Total_Budget_failed_mean")}
- Successful campaigns budget: {driver_stats.get("Total_Budget_success_mean")}
- Failed CPC vs Success CPC: {driver_stats.get("CPC_failed_mean")} vs {driver_stats.get("CPC_success_mean")}

Campaign Setup:
- Failed duration vs Success duration: {driver_stats.get("duration_days_failed_mean")} vs {driver_stats.get("duration_days_success_mean")}
- Failed dominant platform: {driver_stats.get("Platform_Name_failed_top")}
- Successful dominant platform: {driver_stats.get("Platform_Name_success_top")}

Targeting:
- Failed age group: {driver_stats.get("Age_Group_failed_top")}
- Successful age group: {driver_stats.get("Age_Group_success_top")}
- Failed objective: {driver_stats.get("Objective_failed_top")}
- Successful objective: {driver_stats.get("Objective_success_top")}

Tasks:
1. LANGUAGE: Write the entire response in Professional Arabic (Egyptian Business Dialect).
2. Identify the most likely ROOT CAUSES of campaign failures.
3. Explain which setup or targeting decisions are hurting performance.
4. Provide strategic recommendations.

Do NOT mention raw numbers.
Focus on causal reasoning and business strategy.
"""

#senario2
def inflight_summary_prompt(stats: dict) -> str:
    return f"""
You are a senior marketing performance strategist.

We analyzed a set of in-flight marketing campaigns.

Summary:
- Total campaigns: {stats['total_campaigns']}
- Keep running: {stats['keep']}
- Require optimization: {stats['optimize']}
- Should be stopped: {stats['stop']}
- Average performance score: {stats['avg_score']}
- Average expected final ROI: {stats['avg_expected_final_roi']}

Task:
1. LANGUAGE: Write the entire response in Professional Arabic (Egyptian Business Dialect).
2. Provide a concise explanation of the main performance issues.
3. recommend clear optimization actions.
4. Avoid technical jargon and do not mention data sources.
"""
#senario3
def pre_campaign_prediction_prompt(inputs: dict, results: dict) -> str:
    return f"""
    ROLE: Senior Digital Marketing Strategist (15+ years experience).
    TASK: Analyze PRE-CAMPAIGN predictive data from an XGBoost model.
    
    CAMPAIGN CONTEXT:
    - Platform: {inputs['Platform_Name']}
    - Business Goal: {inputs['Objective']}
    - Target Audience: {inputs['Gender']} | Age: {inputs['Age_Group']}
    - Investment: ${inputs['Total_Budget']} for {inputs['duration_days']} days.

    PREDICTED DATA:
    - ROI: {results['ROI']}
    - CTR: {results['CTR']}%
    - CPA: ${results['CPA']}
    - Conversion Rate: {results['Conversion_Rate']}%

    OUTPUT INSTRUCTIONS:
    1. LANGUAGE: Write the entire response in Professional Arabic (Egyptian Business Dialect).
    2. STYLE: Direct, expert-level, and actionable. No repetitive numbers.
    3. STRUCTURE:
       - Final Decision: (GO or NO-GO) with a brief justification.
       - Critical Analysis: Explain what the high CTR vs Conversion Rate means for this campaign.
       - Strategic Pivot: One specific recommendation to maximize the ROI.
       - Budget Feedback: Is the daily spend optimal?
    
    STRICT RULE: Do NOT start with "Greetings" or "Bismillah". Start directly with the analysis.
    """

