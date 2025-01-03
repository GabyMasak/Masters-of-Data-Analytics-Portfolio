#Gaby Masak
#D603 - Machine Learning
#Task 2

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Define the file path
file_path = (r"C:\Users\gabri\OneDrive\Documents\Education\WGU\MSDA\D603 - "
             r"Machine Learning\D603 Task 1\medical_clean.csv")

# Read the CSV file into a DataFrame
medData = pd.read_csv(file_path)

# Drop duplicate rows
medData.drop_duplicates(inplace=True)

# Only continuous data
medData = medData[['Income', 'Initial_days', 'TotalCharge', 'Additional_charges']]

# Handle missing values
for column in medData.columns:
    medData[column] = medData[column].fillna(medData[column].mean())

# Standardize the data
scaler = StandardScaler()
medData_scaled = scaler.fit_transform(medData)

# Export the cleaned and preprocessed data to a CSV file
cleaned_data = pd.DataFrame(medData_scaled)
cleaned_data.to_csv('cleaned_data.csv', index=False)

# Determine the optimal number of clusters using the silhouette method
silhouette_scores = []
for k in range(2, 50):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(medData_scaled)
    score = silhouette_score(medData_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plot the silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(range(2, 50), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method for Determining Optimal Number of Clusters')
plt.show()

# Print the optimal number of clusters
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal number of clusters: {optimal_clusters}")

# Perform k-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
medData['Cluster'] = kmeans.fit_predict(medData_scaled)

# Apply PCA for visualization

pca = PCA(n_components=2)
medData_pca = pca.fit_transform(medData_scaled)

# Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=medData_pca[:, 0], y=medData_pca[:, 1], hue=medData['Cluster'], palette='viridis')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.title('K-Means Clustering with PCA')
plt.show()

# Analyze the clusters
cluster_sizes = medData['Cluster'].value_counts()
print(f"Cluster Sizes:\n{cluster_sizes}")

# Assess cluster quality using Davies-Bouldin Index
db_index = davies_bouldin_score(medData_scaled, medData['Cluster'])
print(f"Davies-Bouldin Index: {db_index}")

# Interpret the clusters
for cluster in range(optimal_clusters):
    cluster_data = medData[medData['Cluster'] == cluster]
    print(f"\nCluster {cluster + 1} Summary:")
    print(cluster_data.describe())

# Unnecessary--just for show.
# Perform hierarchical clustering
Z = linkage(medData_scaled, method='ward')

# Determine the optimal number of clusters via threshold distance
threshold = 100
medData['Cluster'] = fcluster(Z, threshold, criterion='distance')

# Print the number of clusters found
optimal_clusters = len(set(medData['Cluster']))
print(f"Hierarchical: Optimal number of clusters: {optimal_clusters}")

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
