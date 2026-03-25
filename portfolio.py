import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
import config

def select_portfolio_max_sharpe(embeddings_df, features_dict, start_date, end_date):
    """Clusters asset embeddings and allocates weights based on max Sharpe ratios."""
    metrics_list = []
    
    for t in embeddings_df.index:
        pkg = features_dict[t]
        raw_ret, idx = pkg['raw_returns'], pkg['index']
        mask = (idx >= start_date) & (idx < end_date)
        period_ret = raw_ret[mask]
        
        if len(period_ret) < 20: continue
        ann_ret = period_ret.mean() * 252
        ann_vol = period_ret.std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-6)
        metrics_list.append({'Ticker': t, 'Sharpe': sharpe, 'Vol': ann_vol})
        
    metrics_df = pd.DataFrame(metrics_list).set_index('Ticker')
    common = metrics_df.index.intersection(embeddings_df.index)
    metrics_df, embeddings_df = metrics_df.loc[common], embeddings_df.loc[common]
    
    if len(metrics_df) < config.TARGET_CLUSTERS: return pd.DataFrame()
    
    # Agglomerative Clustering Phase
    X_norm = normalize(embeddings_df.values)
    clustering = AgglomerativeClustering(n_clusters=config.TARGET_CLUSTERS, metric='euclidean', linkage='ward')
    metrics_df['Cluster'] = clustering.fit_predict(X_norm)
    
    selected_assets = []
    for i in range(config.TARGET_CLUSTERS):
        group = metrics_df[metrics_df['Cluster'] == i]
        valid_group = group[group['Sharpe'] > 0]
        if not valid_group.empty:
            selected_assets.append(valid_group['Sharpe'].idxmax())
            
    if not selected_assets: return pd.DataFrame()

    final_df = metrics_df.loc[selected_assets].copy()
    
    # Softmax Weight Allocation
    sharpe_vals = torch.tensor(final_df['Sharpe'].values / 0.5) 
    final_df['Weight'] = F.softmax(sharpe_vals, dim=0).numpy()
    
    return final_df[['Weight', 'Sharpe']]
