# Spotify Audio Features Clustering

This project applies clustering techniques to a dataset of Spotify tracks using various audio features such as danceability, energy, tempo, and more. The goal is to uncover natural groupings of songs based on their sound profiles.

## Dataset
The dataset contains thousands of Spotify tracks with the following audio features:
- Acousticness  
- Danceability  
- Duration (ms)  
- Energy  
- Instrumentalness  
- Liveness  
- Speechiness  
- Tempo  
- Valence

Source: [Kaggle Dataset - Spotify Audio Features](https://www.kaggle.com/datasets)

## Methods
- **Data Cleaning**: Removed missing values
- **Standardization**: Used `StandardScaler` to normalize feature values
- **Clustering**: Applied KMeans clustering algorithm
- **Dimensionality Reduction**: Used PCA for visualization (if enabled)
- **Evaluation**: Used the Elbow Method to choose the optimal number of clusters

## Files
- `SpotifyAudioFeaturesClustering.py`: Main Python script for analysis
- `SpotifyAudioFeaturesApril2019.csv`: Dataset file
- `elbow_plot.png`: Visual showing inertia vs. number of clusters
