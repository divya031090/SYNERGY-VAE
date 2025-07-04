import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

encoded_combined = np.load('results/encoded_multidecoder.npy')
df_demo = pd.read_csv('data/raw/Demo_numeric_NHANES_2005_2018.csv')

for k in [2, 3, 4]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(encoded_combined)
    df_demo[f'Cluster_Labels_{k}'] = labels

df_demo[['SEQN', 'Cluster_Labels_2', 'Cluster_Labels_3', 'Cluster_Labels_4']].to_csv(
    'results/Cluster_labels.csv', index=False
)

print("✅ Cluster labels saved.")
