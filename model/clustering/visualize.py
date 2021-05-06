import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

def _visualize_silhouette(X, n_clusters = 3, cluster_labels = None, scoring_method='silhouette', metric = "euclidean"):
	if cluster_labels is None:
		cluster_labels = compute_clusters(X, num_clusters = n_clusters)
	score, score_avg = None, None
	if scoring_method =='silhouette':
		score = silhouette_samples(X, cluster_labels, metric = metric)
		score_avg = silhouette_score(X, cluster_labels, metric = metric)
	print("For n_clusters =", n_clusters, "the average %s score is : %.4f" % (scoring_method, score_avg))
	fig, ax1 = plt.subplots(1, 1)
	fig.set_size_inches(8, 3)
	ax1.set_xlim([-0.1, 1])
	ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
	y_lower = 10
	for i in range(n_clusters):
		ith_cluster_silhouette_values = score[cluster_labels == i]
		ith_cluster_silhouette_values.sort()
		size_cluster_i = ith_cluster_silhouette_values.shape[0]
		y_upper = y_lower + size_cluster_i
		color = cm.nipy_spectral(float(i) / n_clusters)
		ax1.fill_betweenx(np.arange(y_lower, y_upper),
						0, ith_cluster_silhouette_values,
						facecolor=color, edgecolor=color, alpha=0.7)
		ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
		y_lower = y_upper + 10  # 10 for the 0 samples
	ax1.set_title("Plot for the various clusters.")
	ax1.set_xlabel("The % score" % scoring_method)
	ax1.set_ylabel("Cluster label")
	ax1.axvline(x=score_avg, color="red", linestyle="--")
	ax1.set_yticks([])  # Clear the yaxis labels / ticks
	ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
	plt.show()
	return score

def visualize_clusters(data, clusters, pair_plot = False, visualize_tsne = False, visualize_pca = False):
	# Silhouette
	n_clusters = len(np.unique(clusters))
	silhouette_score = _visualize_silhouette(data, n_clusters = n_clusters, cluster_labels=clusters)

	# Visualize num size
	cluster_sizes = pd.Series(clusters).value_counts()
	print(cluster_sizes)  
	plt.hist(clusters)
	plt.show()

	# Pair Plot 
	if pair_plot == True:
		data_clustered = get_data_clustered(data, clusters)
		sns.pairplot(data_clustered, hue = 'ClusterID', palette='Accent')
		plt.show()

	# t-SNE 2D
	if visualize_tsne == True:
		t_SNE_2D(data, clusters)
	if visualize_pca == True:
		pca_2D(data, clusters)
	return cluster_sizes, silhouette_score


''' back-up code ''' 
def elbow_base(data, type_model = "kmeans", covariance_type = 'full', precompute_distances = "auto"):
	scores = []
	list_k = list(range(1, 10))
	for k in list_k:
			model = KMeans(n_clusters=k, precompute_distances = precompute_distances)
			if type_model == 'gmm':
				model = GaussianMixture(n_components=k, covariance_type=covariance_type)
			model.fit(data)
			scores.append(-model.score(data))
	# Plot sse against k
	plt.figure(figsize=(6, 6))
	plt.plot(list_k, scores, '-o')
	plt.title("Elbow "+ type_model.upper())
	plt.xlabel(r'Number of clusters *k*')
	plt.ylabel('Sum of squared distance');
	plt.show()

def elbow_dbscan(data, n_neighbors = 10):
	nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors)
	neighbors = nearest_neighbors.fit(data)
	distances, indices = neighbors.kneighbors(data)
	distances = np.sort(distances[:,n_neighbors-1], axis=0)
	i = np.arange(len(distances))
	knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
	fig = plt.figure(figsize=(5, 5))
	knee.plot_knee()
	plt.title("Elbow DBSCAN")
	plt.xlabel("Points")
	plt.ylabel("Distance")
	plt.show()
	print("found eps =",distances[knee.knee])
def elbow(data, covariance_type="full", n_neighbors=10, precompute_distances="auto"):
	elbow_base(data, type_model= "kmeans", precompute_distances = precompute_distances)
	elbow_base(data, type_model= "gmm", covariance_type=covariance_type)
	elbow_dbscan(data, n_neighbors=n_neighbors)

# Vilualization in 2D
def t_SNE_2D(data, clusters, perplexity = 50):
	#T-SNE with two dimensions
	tsne_2d = TSNE(n_components=2, perplexity=perplexity)
	#This DataFrame contains two dimensions, built by T-SNE
	TCs_2d = pd.DataFrame(tsne_2d.fit_transform(data))
	TCs_2d.columns = ["TC1_2d","TC2_2d"]
	data_clustered = get_data_clustered(data, clusters)  
	plotX = pd.concat([data_clustered,TCs_2d], axis=1, join='inner')
	cluster_list = {}
	for id_cluster in np.unique(clusters):
		cluster = plotX[plotX["ClusterID"] == id_cluster]
		cluster_list[id_cluster] = cluster
	color_list = [ 'rgba(255, 128, 255, 0.8)', 'rgba(255, 128, 2, 0.8)', 'rgba(0, 255, 200, 0.8)', 'brown', 'pink', 'red', 'navy', 'purple', 'black']
	data = []
	for i in np.unique(clusters):
		trace = go.Scatter(
												x = cluster_list[i]["TC1_2d"],
												y = cluster_list[i]["TC2_2d"],
												mode = "markers",
												name = "Cluster %s"%i,
												marker = dict(color = color_list[i]),
												text = None)
		data.append(trace)
	title = "Visualizing Clusters in Two Dimensions Using T-SNE (perplexity=" + str(perplexity) + ")"
	layout = dict(title = title,
								xaxis= dict(title= 'TC1',ticklen= 5,zeroline= False),
								yaxis= dict(title= 'TC2',ticklen= 5,zeroline= False))
	fig = dict(data = data, layout = layout)
	iplot(fig)

def pca_2D(data, clusters):
	# Fit PCA to the good data using only two dimensions
	pca = PCA(n_components=2)
	pca.fit(data_user_num)
	# Apply a PCA transformation the good data
	reduced_data = pca.transform(data_user_num)
	# Create a DataFrame for the reduced data
	reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
	predictions = pd.DataFrame(clusters, columns = ['Cluster'])
	plot_data = pd.concat([predictions, reduced_data], axis = 1)
	# Generate the cluster plot
	fig, ax = plt.subplots(figsize = (14,8))
	# Color map
	cmap = cm.get_cmap('gist_rainbow')
	flag = -1 in np.unique(clusters)
	# Color the points based on assigned cluster
	for i, cluster in plot_data.groupby('Cluster'):   
			cluster.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
									color = cmap((i+1)*1.0/((1-flag)+len(np.unique(clusters)))), label = 'Cluster %i'%(i), s=30);
	# Set plot title
	ax.set_title("Cluster Learning on PCA-Reduced Data");
	plt.show()

def test_density_based(data, type_model = "hdbscan", metric='euclidean', eps=0.3, min_samples=80, min_cluster_size = 10, get_clusters = False, pair_plot = False, visualize_tsne = False, visualize_outlier = False, visualize_pca = False):
	model = hdbscan.HDBSCAN(metric=metric, min_cluster_size = min_cluster_size)
	if type_model == "dbscan":
		model = DBSCAN(eps=eps,min_samples=min_samples, metric = metric)
	model.fit(data)
	clusters = model.labels_
	if get_clusters == False:
		# Get data processed 
		data_clustered = get_data_clustered(data, clusters)
		data_processed = data_clustered[data_clustered.ClusterID >= 0]
		# Silhouette
		num_clusters = len(np.unique(clusters[clusters>=0]))
		for i in [num_clusters]:
				visualize_clusters(data_processed.drop('ClusterID', axis=1), n_clusters = i,cluster_labels=data_processed.ClusterID)
		# Visualize num size
		print(pd.Series(clusters).value_counts())  
		plt.hist(clusters)
		plt.show()
		# Pair Plot 
		if pair_plot == True:
			sns.pairplot(data_processed, hue = 'ClusterID', palette='Accent')
			plt.show()
		# Visualize
		if visualize_outlier == False:
			data = data_processed.drop('ClusterID', axis=1)
			clusters = np.array(data_processed['ClusterID'])
		if visualize_tsne == True:
			t_SNE_2D(data, clusters)
		if visualize_pca == True:
			pca_2D(data, clusters)
	return clusters
