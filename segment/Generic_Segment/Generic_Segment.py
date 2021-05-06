import numpy as np
import pandas as pd
import torch
from model import compute_clusters
from model.utils import get_clustersorted, cal_distribution_bygroup, scale_signunit, scale_minmax

from segment.Segment_Data import SegmentData

class Generic_Segment():
	def __init__(self, cluster_df, cuscluster_names = ['KPI1_Clusters', 'KPI2_Clusters'], kpi_names = ['KPI1', 'KPI2'], generic_name = 'Generic_Cluster', feature_impact_weight=1.0, sort_metric = 'mean'):
		self.feature_impact_weight = feature_impact_weight
		self.sort_metric = sort_metric
		self.cluster_df = cluster_df
		self.cuscluster_names = cuscluster_names
		self.kpi_names = kpi_names
		self.generic_name = generic_name

	def _cal_rangebetweencluster(self, mapping_mean_group):
		tmp_mapping = mapping_mean_group.copy()
		tmp_mapping[len(tmp_mapping)] = 1
		tmp_mapping[-1] = 0 
		mapping_range_group = {}
		for i in range(len(mapping_mean_group)):
			mapping_range_group[i] = 0.5 * min(tmp_mapping[i]-tmp_mapping[i-1], tmp_mapping[i+1]-tmp_mapping[i])
		return mapping_range_group        

	def _cal_clusterinfo(self, cluster_name, kpi_name):
		# sort clusters
		clustername_sorted = get_clustersorted(self.cluster_df, cluster_name = cluster_name, valuesorted_name=kpi_name, sort_metric=self.sort_metric)
		clusterpos_mapping = dict(zip((clustername_sorted), range(len(clustername_sorted))))
		# normalize
		cluster_values = self.cluster_df[[cluster_name, kpi_name]]
		cluster_values[kpi_name] = scale_minmax(np.log(cluster_values[kpi_name]))
		# get distribution information of groups: mean, std, range
		mapping = cal_distribution_bygroup(cluster_values, cluster_name, kpi_name)
		mapping_mean, mapping_std = mapping['mean'], mapping['std']
		clustervalue_range = self.cluster_df.apply(lambda r: r[kpi_name] if pd.isna(mapping_std[r[cluster_name]]) else (r[kpi_name]-mapping_mean[r[cluster_name]])/mapping_std[r[cluster_name]], axis=1)
		# compute unit - corresponding values in distribution (mean = 0, std)
		cluster_values['%s_ClusterUnit' % kpi_name] = scale_signunit(clustervalue_range)
		mapping_range_group = self._cal_rangebetweencluster(mapping_mean)
		# cal ClusterReal
		cluster_info = (cluster_values, clusterpos_mapping, clustervalue_range)
		return cluster_info

	def _cal_ClusterReal(self, cluster_info, cluster_name, kpi_name):
		cluster_values, clusterpos_mapping, clustervalue_range = cluster_info
		n_clusters = cluster_values[cluster_name].nunique() 
		#no-range
		pos = cluster_values[cluster_name].apply(lambda r: clusterpos_mapping[r])
		res_norange = cluster_values['%s_ClusterUnit' % kpi_name] + (0.5+pos)*1.0/n_clusters
		#with-range
		pos = np.zeros((n_clusters))
		pos[0] = clustervalue_range[0]/2
		for i in range(1, n_clusters):
			pos[i] = pos[i-1] + clustervalue_range[i]/2
		res_range = cluster_values.apply(lambda r: r['%s_ClusterUnit' % kpi_name] \
									+ pos[int(r[cluster_name])], axis=1)
		#hybrid
		cluster_values['%s_ClusterReal' % kpi_name] = 0.5 * scale_minmax(res_norange) * self.feature_impact_weight + 0.5 * scale_minmax(res_range) * (1 - self.feature_impact_weight) 
		return cluster_values

	def cal_ClusterReal(self, cluster_name, kpi_name):
		cluster_info = self._cal_clusterinfo(cluster_name, kpi_name)
		cluster_values = self._cal_ClusterReal(cluster_info,  cluster_name, kpi_name)    
		return cluster_values 

	def get_ClusterReal(self):
		all_ClusterValues = {}
		for cluster_name, kpi_name in zip(self.cuscluster_names, self.kpi_names):
			all_ClusterValues[kpi_name] = self.cal_ClusterReal(cluster_name, kpi_name)
		return all_ClusterValues      

	def _sort_genericcluster(self, gen_clusters_df, value_name):
		cluster_sorted = gen_clusters_df.groupby([self.generic_name])[value_name].mean().reset_index()
		cluster_sorted['mean_clusters'] = np.mean(cluster_sorted[value_name], axis=1)
		cluster_sorted_name = cluster_sorted.sort_values(by = 'mean_clusters', ascending=1)[self.generic_name]
		clusterpos_mapping = dict(zip(cluster_sorted_name, range(len(cluster_sorted_name))))
		return gen_clusters_df[self.generic_name].apply(lambda r: clusterpos_mapping[r])

	def extract_clusters(self, num_clusters = 9):
		all_ClusterValues = self.get_ClusterReal()
		#clustering
		data = pd.DataFrame()
		for value in self.kpi_names:
			data[value] = all_ClusterValues[value]['%s_ClusterReal' % value]
		generic_clusters = compute_clusters(data, type_model = "kmeans", precompute_distances = 'auto', num_clusters=num_clusters)
		
		#sorting clusters
		gen_clusters_df = self.cluster_df[self.cuscluster_names + self.kpi_names]
		gen_clusters_df[self.generic_name] = generic_clusters
		generic_clusters = self._sort_genericcluster(gen_clusters_df, self.cuscluster_names)
		return generic_clusters, all_ClusterValues
