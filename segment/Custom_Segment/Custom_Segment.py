import numpy as np
import pandas as pd
import torch

from model.utils import _generate_sortedcluster
from model import compute_clusters, visualize_clusters

from segment.Custom_Segment.Segment_Regressor import SegmentRegressor
from segment.Custom_Segment.Segment_PretrainedFeature import Segment_PretrainedFeature
from segment.Segment_Data import SegmentData

PRETRAIN_PERCENTILES = [0.2, 0.5, 0.8]
class Custom_Segment():
	def __init__(self, data_obj):
		self.data_obj = data_obj
		self.seg_regressor = None
		self.pretrained_obj = None 

	def save_regressor(self, path_to_save):
		seg_regressor = self._get_regressor()
		set_regressor.save_regressor(path_to_save)

	def _get_regressor(self, regressor = None):
		if (self.seg_regressor is None) == False:
			return self.seg_regressor

		self.seg_regressor = SegmentRegressor(self.data_obj)
		if regressor is None:
			regressor = self.seg_regressor.get_best_regressor()
		self.seg_regressor.regressor = regressor 
		return self.seg_regressor    
	
	def _get_pretrained(self, percentiles = PRETRAIN_PERCENTILES, regressor = None):
		if (self.pretrained_obj is None) == False:
			return self.pretrained_obj
		self.pretrained_obj = Segment_PretrainedFeature(self._get_regressor(regressor), percentiles)
		return self.pretrained_obj

	def _get_feature(self, fluct_range = 0.05, percentiles = PRETRAIN_PERCENTILES, regressor = None):
		seg_regressor = self._get_regressor(regressor)
		pretrained_model = seg_regressor.regressor
		pretrained_obj = self._get_pretrained(percentiles, regressor)

		feature_matrix, centers, target_score_list = pretrained_obj.compute_distance(fluct_range)
		
		y_pred = pretrained_model.model(self.data_obj.X_scale)
		mask = np.zeros((y_pred.shape[0], len(target_score_list)))
		for g in range(len(target_score_list)):
			mask[:, g] = (y_pred.data.numpy().reshape(-1) > target_score_list[g]) *2 -1 
		feature_matrix = feature_matrix / np.sum(feature_matrix, axis=1).reshape(feature_matrix.shape[0], -1)
		feature_matrix *= mask
		
		return feature_matrix, centers

	def extract_clusters(self, num_clusters=7, fluct_range = 0.1, percentiles = PRETRAIN_PERCENTILES, regressor=None, precompute_distances='auto'):
		feature_matrix, centers = self._get_feature(fluct_range = fluct_range, percentiles = percentiles, regressor = regressor)
		clusters = compute_clusters(feature_matrix, num_clusters=num_clusters, precompute_distances = precompute_distances)
		# sorting clusters
		clusters = _generate_sortedcluster(self.data_obj.data[self.data_obj.target_name], clusters, sort_metric='median')
		return clusters, feature_matrix, centers  

	def visualize_CustomSeg(self, all_matrix_distance, all_clusters, pair_plot = False, visualize_tsne = False, visualize_pca = False):
		for target_name in all_matrix_distance.keys():
			print("================================================================")
			print("Visualize for ", target_name)
			visualize_clusters(all_matrix_distance[target_name], all_clusters[target_name] , pair_plot = pair_plot, visualize_tsne = visualize_tsne, visualize_pca = visualize_pca)