import numpy as np
import pandas as pd
from model import Training_LinearRegression, Training_PoissonRegression, Training_Bayesian_LinearRegression, Training_Bayesian_PoissonRegression
from model import Regression_Evaluation
from propensity_prediction.tasks.base import Model_Base, EnsembleModel_Base, Task_Base, _missingvalue_processing

class LTVPrediction_Base(Model_Base):
	def __init__(self, context, model_config):
		super().__init__(context, model_config)
		self.list_model = {}
		self.idxgroup_dict = {}
		n_group = self.context.num_clusters
		if (model_config is None) == False:
			if model_config['model_name'] == 'LinearRegression':
				self.model = Training_LinearRegression(1, self.n_features, training_config=model_config['training_config'])
			elif model_config['model_name'] == 'PoissonRegression':
				self.model = Training_PoissonRegression(1, self.n_features, training_config=model_config['training_config'])
			elif model_config['model_name'] == 'Bayesian_LinearRegression':
				self.model = Training_Bayesian_LinearRegression(1, self.n_features, training_config=model_config['training_config'])
			elif model_config['model_name'] == 'Bayesian_PoissonRegression':
				self.model = Training_Bayesian_PoissonRegression(1, self.n_features, training_config=model_config['training_config'])
		self.eva_obj = Regression_Evaluation()

	def _get_idxgroup_dict(self, data_df):  
		group_name = self.context.get_group_name()
		group_list = np.array(data_df[group_name].values.ravel())
		for group in set(group_list):
			self.idxgroup_dict[group] = np.where(group_list == group) 
		return self.idxgroup_dict

	def _train(self, X, y, training_batchsize = 100):
		self.model.fit(X, y, training_batchsize = training_batchsize)
		return self

	def _predict(self, X, predicting_batchsize = 100):
		if self.model is None:
			print("Please train model first!!!")
			return None
		preds = self.model.predict(X)
		return preds.reshape(-1) 		

	def _evaluate(self, X, y, predicting_batchsize = 100):
		y_pred = self._predict(X)
		eva = self.eva_obj.evaluate(y, y_pred)
		return eva


class LTVPrediction_EnsembleBase(EnsembleModel_Base, LTVPrediction_Base):
	def __init__(self, context, model_config):
		LTVPrediction_Base.__init__(self, context, model_config)
		EnsembleModel_Base.__init__(self, context, model_config)

	def prepare_npdata(self, data_df, get_label=True, mode = 'training'):
		data_df = _missingvalue_processing(data_df, self.model_base.context, mode = mode)
		self.idxgroup_dict = self._get_idxgroup_dict(data_df)
		for model in self.list_models:
			model.idxgroup_dict = self.idxgroup_dict
		X, y, ids = super().prepare_npdata(data_df, get_label, mode)
		return X, y, ids

	def invalid_process(self, result_arr):
		result_arr[result_arr < 0] = np.nan 
		return result_arr		
	
	def predict(self, data_df, predicting_batchsize = 100, mode = 'testing'):
		pred_ensemble = super().predict(data_df, predicting_batchsize = predicting_batchsize, mode=mode)['predict']
		id_names = self.model_base.context.get_id_names() 
		cols = list(set(pred_ensemble.columns) - set(['results']) - set(id_names))
		return {'model_name': self.model_name, 'predict': pred_ensemble[id_names + ['results'] + cols]}

class LTVPrediction_TaskBase(LTVPrediction_EnsembleBase, Task_Base):
	def __init__(self, context, model_config):
		super().__init__(context, model_config)
		self._create_sub_models(model_config, LTVPrediction_Base, passing_config = [])
