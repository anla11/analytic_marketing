import numpy as np
import pandas as pd
import torch
from model.utils import torch_scale_minmax

class SegmentData():
	def __init__(self, data, target_name, feature_cols, log_mode = True):
		self.data = data
		self.feature_cols, self.target_name = feature_cols, target_name
		self.X_scale, self.y_scale = None, None 
		self.X_range, self.y_range = None, None
		self.X, self.y = None, None
		(self.X, self.y), (self.X_scale, self.y_scale), self.X_range, self.y_range = self.__get_data_scaled(log_mode)

	def __get_data_scaled(self, log_mode):
		X_df = self.data[self.feature_cols]
		y_df = self.data[self.target_name]
		X = torch.from_numpy(np.array(X_df)).float()
		y = torch.from_numpy(np.array(y_df)).float()

		if log_mode==True:
			X = torch.log(X+1e-18)
			y = torch.log(y+1e-18)
		y_scale, y_min, y_max = torch_scale_minmax(y.reshape(-1, 1))
		X_scale, X_min, X_max = torch_scale_minmax(X)
		return (X, y), (X_scale, y_scale), (X_min, X_max), (y_min, y_max)        
