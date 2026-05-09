import numpy as np
from sklearn.decomposition import PCA

def compute_pca_weights(df):
    """
    Compute feature importance using PCA (unsupervised)
    """
    pca = PCA(n_components=1)
    pca.fit(df)

    importance = abs(pca.components_[0])
    weights = importance / importance.sum()

    return dict(zip(df.columns, weights))
