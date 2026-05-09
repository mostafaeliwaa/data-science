import pandas as pd

def classify(scores: pd.Series):
    q75 = scores.quantile(0.75)
    q40 = scores.quantile(0.40)

    result = pd.Series("failed", index=scores.index)

    result[scores >= q40] = "average"
    result[scores >= q75] = "successful"

    return result
