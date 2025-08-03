# Spotify Audio Features Clustering
# Author: Andrew Hashoush

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("/Users/andrewhashoush/Downloads/archive (3)/SpotifyAudioFeaturesApril2019.csv")


# Select features for clustering
features = ['acousticness', 'danceability', 'duration_ms', 'energy',
            'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']

df_selected = df[features].dropna()

# Standardize the data
scaler = StandardScaler()
scaled = scaler.fit_transform(df_selected)

# Determine optimal number of clusters with Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.tight_layout()
plt.show()

# Fit KMeans with chosen K (ex: 4)
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled)

# Reduce dimensions for visualization
pca = PCA(n_components=2)
components = pca.fit_transform(scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=df['cluster'], palette='tab10')
plt.title("Spotify Song Clusters (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()