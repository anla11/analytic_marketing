import numpy as np
import pandas as pd
import torch
from model.utils import _get_groupdata
from model.predicting.regression.input_trainer import train_input
from model import compute_distance_matrix

from segment.Custom_Segment.Segment_Regressor import SegmentRegressor
from segment.Segment_Data import SegmentData

PRETRAIN_PERCENTILES = [0.2, 0.5, 0.8]
class Segment_PretrainedFeature():
	def __init__(self, segment_regressor, percentiles = PRETRAIN_PERCENTILES):
		# assert (segment_regressor is None)
		self.segment_regressor = segment_regressor 
		self.segment_regressor.regressor.model.eval()

		self.y = self.segment_regressor.segdata_obj.y.data.numpy() 
		self.y_group = _get_groupdata(self.segment_regressor.segdata_obj.data[self.segment_regressor.segdata_obj.target_name], percentiles = percentiles)
		self.X_range = (self.segment_regressor.segdata_obj.X_range[0].data.numpy(), self.segment_regressor.segdata_obj.X_range[1].data.numpy())
		self.y_range = (self.segment_regressor.segdata_obj.y_range[0].data.numpy(), self.segment_regressor.segdata_obj.y_range[1].data.numpy())

	def __get_target_score(self):
		target_df = pd.DataFrame({"Target":self.y, "Target_Group": self.y_group})
		target_group = np.array(target_df.groupby("Target_Group").mean())
		target_score_list = target_group

		target_min, target_max = self.y_range[0], self.y_range[1]
		target_score_list = (target_score_list - target_min) / (target_max - target_min)
		return target_score_list

	def _get_centers(self, target_score, epochs, fluct_range):
		return train_input(self.segment_regressor.regressor.model, target_score, self.segment_regressor.X_scale.shape[1], epochs = epochs, fluct_range = fluct_range).data.numpy()

	def get_centers(self, target_score_list, epochs = 100, is_reversed = True, fluct_range=0.05, sample_center_size= 10):
		print ("* Find centroids for each group of the target")
		centers = []
		for target_score in target_score_list:
			print ("    + Train input for group ", target_score)
			list_target_centers = [self._get_centers(target_score, epochs, fluct_range) for i in range(sample_center_size)]
			center = np.mean(list_target_centers, axis=0)
			centers.append(center)
		if is_reversed == True:
			X_min, X_max = self.X_range[0], self.X_range[1]  
			centers = [(centers[i]*(X_max - X_min)+ X_min) for i in range(len(centers))] 
		return centers     

	def compute_distance_fromcenters(self, centers):
		matrix_distance = compute_distance_matrix(self.segment_regressor.segdata_obj.X.data.numpy(), centers)
		return matrix_distance

	def compute_distance(self, fluct_range=0.05):
		# Get centers reversed
		target_score_list = self.__get_target_score()
		centers = self.get_centers(target_score_list, epochs = 100, is_reversed = True, fluct_range = fluct_range)
		# Compute matrix distance
		matrix_distance = self.compute_distance_fromcenters(centers)
		return matrix_distance, centers, target_score_list  