import numpy as np
#senario 1
def compute_success_score(df, weights):
    score = np.zeros(len(df))

    for feature, w in weights.items():
        score += df[feature] * w

    return score
#senario 2
def performance_score(roi_norm, ctr_norm, conversion_norm):
    return (roi_norm + ctr_norm + conversion_norm) / 3


def pace_score(budget_spent_ratio, time_elapsed_ratio):
    return 1 - abs(budget_spent_ratio - time_elapsed_ratio)


def potential_score(time_elapsed_ratio):
    return 1 - time_elapsed_ratio
