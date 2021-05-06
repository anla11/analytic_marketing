import numpy as np
import pandas as pd
from sklearn.cluster import KMeans 
from sklearn.mixture import GaussianMixture

def get_data_clustered(data, clusters):
	data_clustered = pd.concat([data, pd.Series(clusters)], axis = 1)
	data_clustered.columns = list(data.columns) + ['ClusterID']
	return data_clustered

def _clustering_kmeans(data, num_clusters = 3, precompute_distances = "auto"):
	kmeans = KMeans(n_clusters = num_clusters, precompute_distances = precompute_distances)
	kmeans.fit(data)
	return kmeans

def _clustering_gmm(data, num_clusters = 3, covariance_type = 'full'):
	gmm = GaussianMixture(n_components=num_clusters, covariance_type=covariance_type)
	gmm.fit(data)
	return gmm

def compute_clusters(data, type_model = "kmeans", precompute_distances = 'auto', num_clusters=3):
	model = None
	if type_model == "kmeans":
		model = _clustering_kmeans(data,num_clusters=num_clusters, precompute_distances = precompute_distances)
	if type_model == "gmm":
		model = _clustering_gmm(data, num_clusters=num_clusters)
	clusters = model.predict(data)
	return clusters

def compute_distance_matrix(matrix, centers):
	n_samples = len(matrix)
	n_clusters = len(centers)
	matrix_distance = np.zeros((n_samples, n_clusters))
	for i in range(n_clusters):
		distances = np.sqrt(np.sum(np.power((matrix - centers[i]),2), axis = 1))
		matrix_distance[: , i] = distances
	return matrix_distance    