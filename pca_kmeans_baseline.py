import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X_combined = np.load('data/processed/X_combined.npy')

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_combined)

# Elbow plot
inertia = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_pca)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(K_range, inertia, marker='o')
plt.title("PCA + KMeans Elbow")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()

# Silhouette
sil_scores = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_pca)
    score = silhouette_score(X_pca, kmeans.labels_)
    sil_scores.append(score)

plt.figure()
plt.plot(K_range, sil_scores, marker='o')
plt.title("PCA + KMeans Silhouette")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.show()

print("✅ PCA baseline clustering analysis done.")
