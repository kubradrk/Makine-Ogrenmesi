import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Zaman serisi veri oluştur
np.random.seed(0)
n_samples = 300
t = np.linspace(0, 30, n_samples)
X = np.sin(t[:, np.newaxis]) + np.random.normal(0, 0.3, (n_samples, 1))

# Veriyi standartlaştır
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means modelini oluştur
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

# Modeli eğit
kmeans.fit(X_scaled)

# Küme merkezlerini ve veriyi çiz
plt.figure(figsize=(10, 6))
plt.scatter(t, X_scaled, c=kmeans.labels_, cmap='viridis', s=50, alpha=0.5)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(t[:len(cluster_centers)], cluster_centers, c='red', s=200, marker='x')
plt.xlabel("Zaman")
plt.ylabel("Değer")
plt.title("Zaman Serisi Kümeleme")
plt.show()