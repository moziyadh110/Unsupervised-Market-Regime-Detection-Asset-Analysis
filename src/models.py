"""
Machine learning models for clustering and regime identification.
Implements K-means, Gaussian Mixture Model, and Hidden Markov Model.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def standardize_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize features using StandardScaler.
    
    Args:
        X: Feature DataFrame or array
        
    Returns:
        Tuple of (scaled features, scaler object)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✓ Standardized features: mean=0, std=1")
    return X_scaled, scaler


def find_optimal_k(X: np.ndarray, k_range: range = range(2, 11), metric: str = 'silhouette') -> Tuple[int, list]:
    """
    Find optimal number of clusters using silhouette score or inertia.
    
    Args:
        X: Feature array (should be standardized)
        k_range: Range of k values to test
        metric: 'silhouette' or 'inertia'
        
    Returns:
        Tuple of (optimal_k, scores_list)
    """
    scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        if metric == 'silhouette':
            score = silhouette_score(X, labels)
        else:  # inertia
            score = kmeans.inertia_
        
        scores.append(score)
        print(f"  k={k}: {metric}={score:.4f}")
    
    if metric == 'silhouette':
        optimal_k = k_range[np.argmax(scores)]
    else:  # inertia - lower is better, but we're looking for elbow
        optimal_k = k_range[0]  # default to first in range
    
    print(f"\n✓ Optimal k={optimal_k} (using {metric})")
    return optimal_k, scores


def train_kmeans(X: np.ndarray, n_clusters: int = 4, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    """
    Train K-means clustering model.
    
    Args:
        X: Feature array (should be standardized)
        n_clusters: Number of clusters
        random_state: Random seed
        
    Returns:
        Tuple of (cluster labels, fitted model)
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    
    silhouette = silhouette_score(X, labels)
    
    print(f"✓ K-means trained")
    print(f"  Clusters: {n_clusters}")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Inertia: {model.inertia_:.2f}")
    
    return labels, model


def train_gmm(X: np.ndarray, n_components: int = 4, random_state: int = 42) -> Tuple[np.ndarray, GaussianMixture]:
    """
    Train Gaussian Mixture Model.
    
    Args:
        X: Feature array (should be standardized)
        n_components: Number of mixture components
        random_state: Random seed
        
    Returns:
        Tuple of (cluster labels, fitted model)
    """
    model = GaussianMixture(n_components=n_components, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    
    silhouette = silhouette_score(X, labels)
    bic = model.bic(X)
    
    print(f"✓ Gaussian Mixture Model trained")
    print(f"  Components: {n_components}")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  BIC: {bic:.2f}")
    
    return labels, model


def train_hmm(X: np.ndarray, n_states: int = 4, random_state: int = 42) -> Tuple[np.ndarray, object]:
    """
    Train Hidden Markov Model using hmmlearn.
    
    Args:
        X: Feature array (should be standardized and sorted by time)
        n_states: Number of hidden states
        random_state: Random seed
        
    Returns:
        Tuple of (hidden states, fitted model)
    """
    try:
        from hmmlearn import hmm
        
        # HMM needs data shaped as (n_samples, n_features)
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
        model.fit(X)
        states = model.predict(X)
        
        # Note: HMM doesn't have silhouette score in the same way
        print(f"✓ Hidden Markov Model trained")
        print(f"  States: {n_states}")
        print(f"  Log-likelihood: {model.score(X):.4f}")
        
        return states, model
    except ImportError:
        print("✗ hmmlearn not installed. Run: pip install hmmlearn")
        return None, None


def evaluate_clustering(labels: np.ndarray, X: np.ndarray) -> Dict:
    """
    Evaluate clustering quality.
    
    Args:
        labels: Cluster labels
        X: Feature array
        
    Returns:
        Dictionary with evaluation metrics
    """
    silhouette = silhouette_score(X, labels)
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    
    results = {
        'silhouette_score': silhouette,
        'n_clusters': len(unique),
        'cluster_sizes': dict(zip(unique, counts))
    }
    
    print(f"\n✓ Clustering Evaluation:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Cluster distribution: {dict(zip(unique, counts))}")
    
    return results


def assign_clusters_to_df(df: pd.DataFrame, labels: np.ndarray, column_name: str = 'Regime') -> pd.DataFrame:
    """
    Add cluster labels to original DataFrame.
    
    Args:
        df: Original DataFrame (must have same length as labels)
        labels: Cluster labels
        column_name: Name for the new column
        
    Returns:
        DataFrame with cluster assignments
    """
    df_copy = df.copy()
    df_copy[column_name] = labels
    print(f"✓ Added '{column_name}' column to DataFrame")
    return df_copy
