import numpy as np
import pandas as pd
import torch

from model import Training_Bayesian_LogisticRegression, Training_LogisticRegression
from model.feature_processing.feature_engineering import Feature_Engineering
from model.post_processing.binarize_lib import Binarize
from model import BinaryClassification_Evaluation

from propensity_prediction.tasks.base import Model_Base, EnsembleModel_Base, Task_Base

class ChurnPrediction_Base(Model_Base):
	def __init__(self, context, model_config):
		super().__init__(context, model_config)

		if (model_config is None) == False:
			if model_config['model_name'] == 'LogisticRegression':
				self.model = Training_LogisticRegression(1, self.n_features, training_config=model_config['training_config'])
			if model_config['model_name'] == 'Bayesian_LogisticRegression':
				self.model = Training_Bayesian_LogisticRegression(1, self.n_features, training_config=model_config['training_config'])
		
		self.training_constant['churn_rate'] = None
		self.binarize = Binarize(model_config['binarize_config'])
		self.eva_obj = BinaryClassification_Evaluation(model_config['binarize_config'])

	def _train(self, X, y, training_batchsize = 100):
		if (self.training_constant['churn_rate'] is None):
			if y is None:
				print ('No values of labels')
			self.training_constant['churn_rate'] = np.mean(y)
		self.model.fit(X, y, training_batchsize = training_batchsize)
		return self 

	def _get_probabilities(self, X, predicting_batchsize = 100):
		prob = self.model.predict_proba(X, predicting_batchsize = predicting_batchsize)[:, 1]
		return prob.reshape(-1)

	def _predict(self, X, predicting_batchsize = 100):
		prob = self._get_probabilities(X, predicting_batchsize = predicting_batchsize)[:, 1].reshape(-1)
		if (self.binarize.binarize_config['method'] == 'gettop'):
			ntop = int(len(X)*self.training_constant['churn_rate'])
			self.binarize.update_config('ntop', ntop)
		predict = self.binarize.binarize(prob)
		return predict.reshape(-1)
		
	def _evaluate(self, X, y, predicting_batchsize = 100):
		prob = self._get_probabilities(X, predicting_batchsize= predicting_batchsize)
		eva = self.eva_obj.evaluate(y, prob)
		return eva 

class ChurnPrediction_EmsembleBase(EnsembleModel_Base, ChurnPrediction_Base):
	def __init__(self, context, model_config):
		ChurnPrediction_Base.__init__(self, context, model_config)
		EnsembleModel_Base.__init__(self, context, model_config)

	def prepare_npdata(self, data_df, get_label=True, is_tensor=False, mode = 'training'):
		X, y, ids = super().prepare_npdata(data_df, get_label, mode = mode)
		if (self.get_trainingconstant('churn_rate') is None) and (mode == 'training'):
			if y is None:
				print ('No values of labels')
			self.update_trainingconstant('churn_rate', np.mean(y))
		if is_tensor:
			X, y = self._convert_totensor(X, y)
		return X, y, ids		

	def predict(self, data_df, predicting_batchsize = 100):
		prob_ensemble = self.get_probabilities(data_df, predicting_batchsize = predicting_batchsize)['probabilities']
		prob_ensemble.rename(columns = {'results': 'probabilities'}, inplace = True)
		
		if (self.binarize.binarize_config['method']=='gettop') and (self.binarize.binarize_config['ntop'] is None):
			ntop = int(len(data_df)*self.get_trainingconstant('churn_rate'))
			self.binarize.update_config('ntop', ntop)
		
		prob_ensemble.loc[:, 'predict'] = self.binarize.binarize(np.array(prob_ensemble['probabilities']))
		cols = self.model_base.context.get_id_names() + ['probabilities', 'predict']
		pred_res = {'model_name': self.model_name, 'predict': prob_ensemble[cols]}
		return pred_res

	def evaluate(self, data_df, predicting_batchsize = 100):
		return super().evaluate(data_df, predicting_batchsize, res_func = 'get_probabilities')


class ChurnPrediction_TaskBase(ChurnPrediction_EmsembleBase, Task_Base):
	def __init__(self, context, model_config):
		super().__init__(context, model_config)
		self._create_sub_models(model_config, ChurnPrediction_Base, passing_config = ['binarize_config'])
