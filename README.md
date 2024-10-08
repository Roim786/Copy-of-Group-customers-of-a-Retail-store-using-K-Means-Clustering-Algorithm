# Copy-of-Group-customers-of-a-Retail-store-using-K-Means-Clustering-Algorithm
# Data Cleaning

from google.colab import drive

drive.mount('/content/drive')



import pandas as pd

data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Mall_Customers.csv')

#Check for missing values

print("Missing values before cleaning are: \n", data.isnull().sum())

#Handling missing values(dropping rows with missing vals)

data.dropna(inplace= True)

#Remove duplicate rows

initial_shape = data.shape

data.drop_duplicates(inplace= True)

print(f"Removed duplicates rows are : {initial_shape[0]-data.shape[0]}")

Missing values before cleaning are: 

 CustomerID                0
Gender                    0
Age                       0
Annual Income (k$)        0
Spending Score (1-100)    0
dtype: int64
Removed duplicates rows are : 0

# Finding and Removing Outliers using Interquartile Ranges (IQR)

#finding Q1 and Q3 for IQR

#but IQR is for numerical data

numeric_cols= ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

numeric_data= data[numeric_cols]

Q1= numeric_data.quantile(0.25)

Q3= numeric_data.quantile(0.75)

#print(Q1,Q3)

IQR= Q3-Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

print("\n\n",lower_bound,upper_bound)

#data[] , inside brackets true will be present and false will be removed
#it is taking Logical OR means condition will be true for any data >upperbound or <lowerbound,which shows
#the outliers,but with ~ sign , outliers will become false and rest data will be true so only true in the
#indexes will remain there and false will be removed

outlier_mask = ((numeric_data>upper_bound)|(numeric_data<lower_bound)).any(axis=1)

print("\n\n", data)

#You can check the data it has now 1 to 198 ids,,, 2 ids in which there are outliers already removed from
#original data ,,and now if i try to check outliers they show empty cuz running cell second time and original
#has been changed

data_without_outliers= data[~outlier_mask]

print(f"Outliers are :\n {data[outlier_mask]}")

print(f"Data After Outliers is: \n {data_without_outliers}")



 Age                       -1.625
Annual Income (k$)       -13.250
Spending Score (1-100)   -22.625
dtype: float64 Age                        79.375
Annual Income (k$)        132.750
Spending Score (1-100)    130.375
dtype: float64

# Standardizing the data using Standardscalar

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  #Initializes the scaler

#we will apply Standardscalar to the numeric data of the cleaned data

data = data_without_outliers

data.loc[:, numeric_cols] = scaler.fit_transform(data[numeric_cols])

#Print the modified DataFrame

print(data)


# Finding Optimal K using Elbow Method

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

#Elbow Method does not have built-in function so we does it using K-means
inertia = [] #, inertia means within-cluster sum of squares (WCSS) refers to the measure of how well the data points fit within the clusters. Specifically, it is the sum of squared distances between each data point and the centroid of the cluster to which it belongs. Lower inertia indicates that the data points are closer to their cluster centroids, suggesting more compact clusters.

k_range = range(1,11)

for k in k_range:

  kmeans = KMeans(n_clusters= k, random_state=42)
  
  kmeans.fit_transform(data[numeric_cols])
  
  inertia.append(kmeans.inertia_)
  
plt.plot(k_range, inertia, marker='o')

plt.xlabel('Number of Clusters (K)')

plt.ylabel('Intertia')

plt.title('Elbow Method')

plt.show()


#According to my dataset after checking the graph the optimal no. of K is 4


# Finding Optimal Number of K using Silhouettes Score

from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in k_range[1:]: #Silhouette always starts from 2

  kmeans = KMeans(n_clusters=k, random_state=42)
  
  cluster_labels = kmeans.fit_predict(data[numeric_cols])
  
  silhouette_avg = silhouette_score(data[numeric_cols], cluster_labels)
  
  silhouette_scores.append(silhouette_avg)

plt.plot(k_range[1:], silhouette_scores, marker='o')

plt.xlabel('Number of Clusters (K)')

plt.ylabel('Silhouette Score')

plt.title('Silhouette Score Method')

plt.show()


# Applying K-means clustering

print(data)

optimal_k=5

kmeans = KMeans(n_clusters=optimal_k, random_state=42)

clusters= kmeans.fit_predict(data[numeric_cols])

data['Cluster']= clusters 

#Addind clusters to the original data as a separate col

print(data)

# Dimentionality Reduction using PCA and Plot Clusters (For verification)

from sklearn.decomposition import PCA

pca = PCA(n_components = 2) 

#It means PCA is reducing the dimensions to 2D data for better visualization of clusters

pca_data = pca.fit_transform(data[numeric_cols])

#Plotting clusters

plt.figure(figsize=(10, 7))

scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data['Cluster'], cmap='viridis', marker='o')

plt.colorbar(scatter)

plt.xlabel('PCA Component 1')

plt.ylabel('PCA Component 2')

plt.title('Clusters Visualization after PCA')

plt.grid(True)

plt.show()

#If Clusters are distinct then if will verified that K is optimal

#If silhouettes give different no. of Optimal K than Elbow method check for both K values to ensure which one has better distinction between clusters


# Verifying Clusters

import seaborn as sns

#Visualize clusters to ensure they are distinct

plt.figure(figsize=(10, 7))

sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=data['Cluster'], palette='viridis', marker='o')

plt.xlabel('PCA Component 1')

plt.ylabel('PCA Component 2')

plt.title('Cluster Visualization')

plt.legend(title='Cluster')

plt.grid(True)

plt.show()
