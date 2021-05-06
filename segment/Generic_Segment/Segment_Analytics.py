import numpy as np
import pandas as pd
import torch

from model.utils import save_picklejson, load_picklejson
from model import Training_LinearRegression
from model import visualize_clusters
from segment.visualize_segmentana import visualize_matrixcluster, vis_ClusterReal

from segment.Custom_Segment.Segment_Regressor import SegmentRegressor
from segment.Custom_Segment.Custom_Segment import Custom_Segment
from segment.Segment_Data import SegmentData
from segment.Generic_Segment.Generic_Segment import Generic_Segment

class Segment_Analytics():
	def __init__(self, data, list_targets, list_features, cusseg_targets, generic_name='Generic_Cluster', \
								custom_settings = {'num_clusters': 9, 'fluct_range':0.1, 'path_name_model':None, 'log_mode':True}, \
								generic_settings = {'num_clusters': 9, 'feature_impact_weight': 1.0}):
		self.all_matrix_distance, self.all_clusters, self.all_centers, self.all_models = None, None, None, None
		self.generic_name, self.generic_clusters, self.all_ClusterValues = generic_name, None, None

		self.data = data
		self.list_targets, self.list_features, self.cusseg_targets  = list_targets, list_features, cusseg_targets
		self.custom_settings, self.generic_settings = custom_settings, generic_settings
		self.__custom_segment, self.__generic_segment = Custom_Segment(None), Generic_Segment(None) # for visualization

	''' Generate custom clusters'''

	def _create_segmentdata(self, target_name, log_mode):
		return SegmentData(self.data, target_name, self.list_features, log_mode=log_mode)

	def _generate_CustomCluster(self, target_name, num_clusters=9, fluct_range = 0.1, regressor = None, log_mode = True):#, segregressor_target = None):
		data_obj = self._create_segmentdata(target_name, log_mode)
		customseg_obj = Custom_Segment(data_obj)
		# customseg_obj.seg_regressor = segregressor_target
		clusters, matrix_distance, centers = customseg_obj.extract_clusters(num_clusters, fluct_range, regressor = regressor)
		return clusters, matrix_distance, centers, customseg_obj.seg_regressor.regressor

	# def _get_regressor(self, target_name, log_mode):
	# 	segregressor_target = None
	# 	if ((self.all_models is None) == False):
	# 		if (target_name in self.all_models.keys()):
	# 			segregressor_target = SegmentRegressor(self._create_segmentdata(target_name, log_mode))
	# 			segregressor_target.regressor = self.all_models[target_name]
	# 	return segregressor_target        

	def _generate_AllCustomCluster(self, num_clusters = 9, fluct_range = 0.1, list_regressors = None, log_mode = True):
		if self.all_clusters is None:
			self.all_matrix_distance, self.all_clusters, self.all_centers = {}, {}, {}
			if self.all_models is None:
				self.all_models = {}
			for target_name in self.cusseg_targets:
				# segregressor_target = self._get_regressor(target_name, log_mode)
				regressor = None
				if (list_regressors is None) == False:
					regressor = list_regressors[target_name]
				self.all_clusters['%s' % target_name], self.all_matrix_distance['%s' % target_name], self.all_centers['%s' % target_name], self.all_models['%s' % target_name]= \
						self._generate_CustomCluster(target_name, num_clusters, fluct_range, regressor, log_mode)#, segregressor_target)
		return self.all_matrix_distance, self.all_clusters, self.all_centers, self.all_models       

	def generate_CustomCluster(self):
		num_clusters, fluct_range, path_name_model, log_mode = self.custom_settings['num_clusters'], self.custom_settings['fluct_range'], self.custom_settings['path_name_model'], self.custom_settings['log_mode']
		list_regressors = self.load_regressors(path_name_model)
		return self._generate_AllCustomCluster(num_clusters, fluct_range, list_regressors, log_mode)

	def _generate_clusterdf(self):
		all_matrix_distance, all_clusters, all_centers, all_models  = self.generate_CustomCluster()
		cluster_df = self.data[self.list_targets]# + self.list_features]
		for target in self.cusseg_targets:
			cluster_df['%s_Cluster'%target] = all_clusters['%s'%target]

		cluster_df_log = np.log(self.data[self.list_targets])# + self.list_features])
		for target in self.cusseg_targets:
			cluster_df_log['%s_Cluster'%target] = all_clusters['%s'%target]
		return cluster_df, cluster_df_log

	''' Generate generic clusters'''

	def _generate_GenericSegment(self, num_clusters = 10, feature_impact_weight = 1.0):
		if self.generic_clusters is None:
			cluster_df, _ = self._generate_clusterdf()
			genseg_obj = Generic_Segment(cluster_df,['%s_Cluster' % target for target in self.cusseg_targets], self.cusseg_targets, self.generic_name, feature_impact_weight = feature_impact_weight, sort_metric = 'mean')
			self.generic_clusters, self.all_ClusterValues = genseg_obj.extract_clusters(num_clusters=9)
		return self.generic_clusters, self.all_ClusterValues

	def generate_GenericSegment(self):
		num_clusters, feature_impact_weight = self.generic_settings['num_clusters'], self.generic_settings['feature_impact_weight']
		return self._generate_GenericSegment(num_clusters, feature_impact_weight)

	''' Save '''

	def save_segments(self, customerID = None, path_to_save =''):
		# Requirements: path_to_save need / at the end.
		all_matrix_distance, all_clusters, all_centers, all_models  = self.generate_CustomCluster()
		cluster_df = pd.DataFrame(customerID)
		for target in self.cusseg_targets:
			cluster_df['%s_Cluster'%target] = all_clusters['%s'%target]
		cluster_df[self.generic_name] = self.generate_GenericSegment()[0]
		cluster_df.to_csv('%scluster_df.csv'%path_to_save, index = False) 
		return cluster_df

	def save_regressors(self, path_to_save):
		save_packages = {}
		for name, model in self.all_models.items(): 
			save_packages[name] = model.package_for_save(self.list_features)
		save_picklejson(save_packages, path_to_save)

	def load_regressors(self, path_to_load):
		if path_to_load is None:
			return None
		self.all_models = {}
		load_package = load_picklejson(path_to_load)
		for name, package in load_package.items():
			model = Training_LinearRegression(output_shape=1, input_shape=len(self.list_features))
			model.load_from_savepackage(package, self.list_features)
			self.all_models[name] = model
		return self.all_models

	''' Visualize '''  

	def visualize_CustomSeg(self, pair_plot = False, visualize_tsne = False, visualize_pca = False):
		all_matrix_distance, all_clusters, all_centers, all_models = self.generate_CustomCluster()
		self.__custom_segment.visualize_CustomSeg(all_matrix_distance, all_clusters, pair_plot = pair_plot, visualize_tsne = visualize_tsne, visualize_pca = visualize_pca)

	def visualize_matrixcluster_CustomSeg(self):
		for i in range(len(self.cusseg_targets)):
			for j in range(i+1, len(self.cusseg_targets)):
				kpi_name = (self.cusseg_targets[i], self.cusseg_targets[j])
				print (kpi_name)
				_, cluster_df_log = self._generate_clusterdf()
				cluster_df_log['CustomerID'] = range(len(cluster_df_log))
				visualize_matrixcluster(cluster_df_log.rename(columns={'CustomerID': 'Number of Customers'}), "Number of Customers", generic_group_define=None, func_group_name = "count", func_sum = np.sum, histogram_vis=False, kpi_name = kpi_name)

				for field in self.list_targets:
					if field in cluster_df_log.columns:        
						visualize_matrixcluster(cluster_df_log, field, generic_group_define=None, func_group_name = "mean", func_sum = np.mean, histogram_vis=False, kpi_name = kpi_name)

	def visualize_GenericSegment(self):
		gen_clusters, all_ClusterValues = self.generate_GenericSegment()
		data = pd.DataFrame()
		for target in self.cusseg_targets:
			data[target] = all_ClusterValues[target]['%s_ClusterReal' % target]
		visualize_clusters(data, gen_clusters , pair_plot = False, visualize_tsne = False, visualize_pca = False)

	def visualize_matrixcluster_GenericSeg(self, vis_clusterreal=True):
		_, cluster_df_log = self._generate_clusterdf()
		gen_clusters, all_ClusterValues = self.generate_GenericSegment()
		cluster_df_log[self.generic_name] = gen_clusters+1
		visualize_matrixcluster(cluster_df_log, self.generic_name, generic_group_define=None, func_group_name = "mean", func_sum = np.mean, histogram_vis=False, heatmap_annot=True)
		if vis_clusterreal: 
			for field in self.cusseg_targets:
				vis_ClusterReal(all_ClusterValues[field], field)