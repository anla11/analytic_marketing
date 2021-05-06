import pandas as pd
from model.utils.pdgroup_processing import get_summarizegroup

def get_clustersorted(df, cluster_name = 'TotalPrice_Cluster', valuesorted_name = 'TotalPrice', sort_metric = 'mean'):
	cluster_sorted = get_summarizegroup(df, cluster_name, valuesorted_name, sort_metric=sort_metric).sort_values(by=valuesorted_name)
	return cluster_sorted[cluster_name]

def generate_clusterstargeted(feature_df, target_list, feature_cols, num_clusters = 8, fluct_range = 0.05):
	all_matrix_distance, all_clusters, all_centers, all_model_weights = {}, {}, {}, {}
	for target_name in target_list:        
		all_matrix_distance['%s' % target_name], all_clusters['%s' % target_name], all_centers['%s' % target_name], all_model_weights['%s' % target_name] = extract_cluster_frompretrained(feature_df, target_name, feature_cols,  num_clusters=num_clusters, log_mode=True, fluct_range = fluct_range)
	return all_matrix_distance, all_clusters, all_centers, all_model_weights

def _generate_sortedcluster(values, clusters, sort_metric='mean'):
	data = pd.DataFrame({'value':values, 'cluster':clusters})
	cluster_sorted = get_clustersorted(data, cluster_name = 'cluster', valuesorted_name='value', sort_metric=sort_metric)
	clusterpos_mapping = dict(zip(cluster_sorted, range(len(cluster_sorted))))
	new_clustername = data['cluster'].apply(lambda r: clusterpos_mapping[r])
	return new_clustername

def generate_sortedcluster(cluster_df, target_names, sort_metric='mean'):
	for target in target_names:
		cluster_sorted = get_clustersorted(cluster_df, cluster_name = '%s_Cluster'%target, valuesorted_name=target, sort_metric=sort_metric)
		clusterpos_mapping = dict(zip(cluster_sorted, range(len(cluster_sorted))))
		new_clustername = cluster_df['%s_Cluster' %target].apply(lambda r: clusterpos_mapping[r])
		cluster_df['%s_Cluster' %target] = new_clustername
	return cluster_df