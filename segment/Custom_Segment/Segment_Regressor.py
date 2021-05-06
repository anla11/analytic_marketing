import numpy as np
import pandas as pd
import torch
from model.utils import cal_correlation
from model import Training_LinearRegression
from model.predicting.regression.linear_regression import LinearRegression
from model.predicting.loss_func import KLDiv_Loss, ELBO_Loss, Combining_Loss, KLandMSE_Loss, ELBOandMSE_Loss
from segment.Segment_Data import SegmentData

class SegmentRegressor():
	def __init__(self, segdata_obj, MAX_LOOP=10, THRES=0.8):
		self.segdata_obj = segdata_obj
		self.regressor = None
		self.MAX_LOOP, self.THRES = MAX_LOOP, THRES
		self.X_scale, self.y_scale = segdata_obj.X_scale, segdata_obj.y_scale 
		# self.X_range, self.y_range = segdata_obj.X_range, segdata_obj.y_range 

	def _get_featureimpact(self):
		weights = [cal_correlation(np.array(self.X_scale[:, i]), np.array(self.y_scale).reshape(-1))[0] for i in range(self.X_scale.shape[1])]
		weights /= np.sum(weights)
		return weights

	def _get_initstate(self):
		package = LinearRegression(1, self.X_scale.shape[1])
		weights = self._get_featureimpact()
		package.linear.weight.data[0] = torch.from_numpy(weights)
		return package.state_dict()

	def _init_regressor(self, loss_func = torch.nn.MSELoss(), epochs = 300, lr = 0.001):
		training_config = {'learning_rate':lr, 'epochs':epochs}
		init_regressor = Training_LinearRegression(1, len(self.segdata_obj.feature_cols), loss_func, training_config)
		init_regressor.load_from_savepackage(self._get_initstate(), mode='training')
		return init_regressor

	def train_regressor(self, loss_func = torch.nn.MSELoss(), epochs = 300, lr = 0.001):
		cnt_earlystopping = 0
		while cnt_earlystopping<10:
			self.regressor = self._init_regressor(loss_func, epochs, lr)
			self.regressor.fit(self.X_scale, self.y_scale)
			if self.regressor.early_stopping > 50:
				break
			cnt_earlystopping += 1
		parameter_value = [v.data.numpy() for v in list(self.regressor.model.parameters())]
		return self.regressor

	def get_best_regressor(self, epsilon=0.1):
		print ("* Find the best regressor!")
		res_vmax = None
		self.regressor = None
		for loop in range(self.MAX_LOOP):
			list_loss_func = [torch.nn.MSELoss(), KLDiv_Loss(reduction='mean'), ELBO_Loss(reduction='mean'), Combining_Loss(reduction='mean'), KLandMSE_Loss(reduction='mean'), ELBOandMSE_Loss(reduction='mean')]
			list_regressor = []
			for loss_func in list_loss_func:
				print ("    + Training regressor with", loss_func)
				list_regressor.append(self.train_regressor(loss_func=loss_func, epochs=300))
		
			list_cor = []
			for regressor in list_regressor:
				cor, conf = cal_correlation(self.y_scale.data.numpy().reshape(-1), regressor.model(self.X_scale).data.numpy().reshape(-1))
				list_cor.append(cor)
			base = list_cor[0]
			vmax, idx = np.max(list_cor[1:]), np.argmax(list_cor[1:]) + 1
			best_regressor = None
			if (vmax > self.THRES):
				return list_regressor[idx]
			if vmax - base >= -epsilon:
				best_regressor = list_regressor[idx]
			else:
				best_regressor = list_regressor[0]
				vmax = base
			if (res_vmax is None) or (vmax > res_vmax):
				self.regressor = best_regressor, 
				res_vmax = vmax
		return self.regressor

	def set_regressor(self, regressor):
		self.regressor = regressor