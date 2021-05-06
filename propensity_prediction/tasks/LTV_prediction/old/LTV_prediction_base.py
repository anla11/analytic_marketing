import numpy as np
import pandas as pd
# from sklearn.linear_model import LinearRegression 
from propensity_prediction.model import Training_LinearRegression
from propensity_prediction.utils import r2_score
from propensity_prediction.tasks.base import Model_Base, EnsembleModel_Base, Task_Base

class LTVPrediction_Base(Model_Base):
	def __init__(self, context, model_config):
		super().__init__(context, model_config['model_name'], None, None)

	def _convert_totensor(self, X, y):
		X = torch.tensor(X).float()
		if (y is None) == False:
			y = torch.tensor(y).float()
		return X, y

	def prepare_npdata(self, data_df, get_label=True, is_tensor=False):
		X, y = super().prepare_npdata(data_df, get_label, mode=mode, logging=logging)
		if is_tensor:
			X, y = self._convert_totensor(X, y)
		return X, y

	def _train(self, X, y, training_batchsize = 100):
		# self.model = LinearRegression()
		# self.model.fit(X, y)
		n_features = len(self.context.get_feature_names())
		self.model = Training_LinearRegression(1, n_features)#, training_config=model_config['training_config'])
		self.model.fit(X, y, training_batchsize = training_batchsize)
		return self.model 

	def _get_probabilities(self, X, predicting_batchsize = 100):
		prob = self.model.predict_proba(X, predicting_batchsize = predicting_batchsize)
		return prob		

	def _predict(self, X, predicting_batchsize = 100):
		if self.model is None:
			print("Please train model first!!!")
			return None
		y_pred = self.model.predict(X)
		return y_pred 		

	def _evaluate(self, X, y, predicting_batchsize = 100):
		y_pred = self._predict(X)
		r_square = r2_score(y, y_pred)
		return r_square


class LTVPrediction_EnsembleBase(EnsembleModel_Base, LTVPrediction_Base):
	def __init__(self, context, model_config):
		LTVPrediction_Base.__init__(self, context, model_config)
		EnsembleModel_Base.__init__(self, context, model_config)

	def __predict_ensemble(self, data_df, predicting_batchsize = 100):
		prob_obj = super().predict(data_df, predicting_batchsize= predicting_batchsize)
		prob_list = prob_obj['predict']

		all_model_res = {}
		all_prob = []
		for model_res in prob_list:
			pred_df = model_res['id']
			pred_df['predict_%s' % model_res['model_name']] = model_res['predict']
			all_prob.append(model_res['predict'])
			all_model_res[model_res['model_name']] = pred_df

		all_res_df = None
		for model_name, model_df in all_model_res.items():
			if all_res_df is None:
				all_res_df = model_df.copy()
			else:
				all_res_df = all_res_df.merge(model_df, how = 'outer', on = self.list_models[0].context.get_id_names())
		all_res_df['predict'] = np.mean(all_prob, axis = 0)
		return all_res_df

	def get_probabilities(self, data_df, predicting_batchsize = 100):
		return None 
		
	def predict(self, data_df, predicting_batchsize = 100):
		pred_ensemble = self.__predict_ensemble(data_df, predicting_batchsize = predicting_batchsize)
		cols = self.list_models[0].context.get_id_names() + ['predict']
		return {'model_name': self.model_name, 'predict': pred_ensemble[cols]}		

	def evaluate(self, data_df, predicting_batchsize = 100):
		eva_res = super().evaluate(data_df, predicting_batchsize = predicting_batchsize)
		return {'model_name': self.model_name, 'evaluation': eva_res['evaluation']}

		# eva_res = super().evaluate(data_df, predicting_batchsize = predicting_batchsize)
		# pred_ensemble = self.predict(data_df, predicting_batchsize = predicting_batchsize)['predict']
		# y = self.get_labels(data_df)
		# r_square = r2_score(y, pred_ensemble)
		# return {'model_name': self.model_name, 'evaluation': eva_ensemble, 'evaluate_sub_models': eva_res['evaluation']}		

class LTVPrediction_TaskBase(LTVPrediction_EnsembleBase, Task_Base):
	def __init__(self, context, model_config):
		super().__init__(context, model_config)
		self.list_models = []
		for sub_model_config in model_config['methods']:
			if sub_model_config['model_name'] == "Regression":
				model = LTVPrediction_Base(context, model_config)
				self.list_models.append(model)


