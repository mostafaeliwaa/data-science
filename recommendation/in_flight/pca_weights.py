import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def learn_inflight_pca_weights(score_df: pd.DataFrame):
    """
    score_df columns:
    [
        'performance_score',
        'pace_score',
        'potential_score'
    ]
    """


    score_df = score_df.dropna()
    pca = PCA(n_components=1)
    pca.fit(score_df)
    loadings = np.abs(pca.components_[0])

    
    weights = loadings / loadings.sum()

    return {
        "performance": float(weights[0]),
        "pace": float(weights[1]),
        "potential": float(weights[2]),
        "explained_variance": float(pca.explained_variance_ratio_[0])
    }
