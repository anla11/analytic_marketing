import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from propensity_prediction.tasks.converting_action_prediction.base.context import NextAction_Context
from propensity_prediction.tasks.base import Model_Base, EnsembleModel_Base
from propensity_prediction.utils.evaluation.binaryclass_evaluate import BinaryClassification_Evaluation
from propensity_prediction.model.post_processing.binarize_lib import Binarize

from propensity_prediction.utils.index_process import Batch_Data

class NextAction_Base(Model_Base):
	def __init__(self, context, model_config):
		super().__init__(context, model_config['model_name'], None, BinaryClassification_Evaluation())
		self.multi_class = model_config['multi_class']
    
	def get_labels(self, data_df): #return a numpy array of labels
		labels = self.context.get_label_names()
		y = np.array(data_df[labels[0]].values.ravel()).astype(int)
		# Check if label for multiclass, we will create a new label
		if self.multi_class == True:
			y = np.zeros((len(data_df), ))
			for i, label in enumerate(labels):       
				y = np.where(data_df[label] > 0, i+1, y)
		return np.array(y).astype(int)

	def _train(self, X, y, training_batchsize = 100):
		self.model = LogisticRegression(class_weight='balanced', max_iter=1000)
		
		batch_data_obj = Batch_Data(X.shape[0], training_batchsize)
		while (batch_data_obj.is_end() == False):
			print ('    Training with minibacth ', batch_data_obj.cur_batchidx)
			start_idx, end_idx = batch_data_obj.enum_batch()
			self.model.fit(X[start_idx:end_idx, :], y[start_idx:end_idx])

		return self.model         

	def _get_probabilities(self, X, predicting_batchsize = 100):
		labels = self.context.get_label_names()
		n_classes = 2 if self.multi_class == False else len(labels) + 1
		prob_table = np.zeros((X.shape[0], n_classes))

		batch_data_obj = Batch_Data(X.shape[0], predicting_batchsize)
		batch_data_obj.run(self.model.predict_proba, data = X, outputs = prob_table)	
		# prob_table = self.model.predict_proba(X)
		return prob_table

	def package_for_save(self):
		return self.model

	def load_from_savepackage(self, model):
		self.model = model
		return self			


class NextAction_BinaryClass_Base(NextAction_Base):
	def __init__(self, context, model_config):
		super().__init__(context, model_config)
		self.binarize = Binarize()
		self.binarize_config = model_config['binarize_config']        
		self.dropna_preprocess = model_config['dropna_preprocess']

	def custom_preprocess_inmodel(self, data_df):
		new_df = data_df.copy()
		if self.dropna_preprocess == True:
			for order_action in self.context.order_actions:
				new_df = new_df.drop(new_df[pd.isnull(new_df[order_action['source']]) & pd.notnull(new_df[order_action['des']])].index)
		for i in self.context.get_convfeature_names():
			new_df[i].fillna(new_df[i].mean(),inplace=True)
		new_df = new_df.dropna()
		return new_df

	def _predict(self, X, predicting_batchsize = 100):
		if self.model is None:
			print("Please train model first!!!")
		prob = self._get_probabilities(X, predicting_batchsize = predicting_batchsize)[:,1]
		if (self.binarize_config['method']=='gettop') and (self.binarize_config['ntop'] is None):
			self.binarize_config['ntop'] = int(len(data_df)*self.churn_rate)
		if (self.binarize_config['method']=='threshold') and (self.binarize_config['thres_type'] is None):
			self.binarize_config['thres_type'] = 'baseline'
			
		self.binarize.update_config('method', 'threshold')
		self.binarize.update_config('thres_type', 'baseline')
		predictions = self.binarize.binarize(prob)
		return predictions

	def _evaluate(self, X, y, predicting_batchsize = 100):
		prob = self._get_probabilities(X, predicting_batchsize = predicting_batchsize)[:, 1]
		eva  = self.eva_obj.evaluate(y, prob)
		return eva    


class NextAction_MultiClass_Base(NextAction_Base):
	def __init__(self, context):
		model_config = {'model_name': 'NextAction_MultiClass', 'multi_class':True}        
		super().__init__(context, model_config)

	def custom_preprocess_inmodel(self, data_df):
		new_df = data_df.copy()
		for order_action in self.context.order_actions:
			new_df = new_df.drop(new_df[pd.isnull(new_df[order_action['source']]) & pd.notnull(new_df[order_action['des']])].index)
		for i in self.context.get_convfeature_names():
			new_df[i].fillna(new_df[i].mean(),inplace=True)
		new_df = new_df.dropna()
		return new_df

	def _predict(self, X, predicting_batchsize = 100):
		if self.model is None:
			print("Please train model first!!!")
			return None
		batch_data_obj = Batch_Data(X.shape[0], predicting_batchsize)
		predictions = np.zeros((X.shape[0]))
		batch_data_obj.run(self.model.predict, data = X, outputs = predictions)	

		return predictions 

	def _evaluate(self, X, y, predicting_batchsize = 100):
		prob_all = self._get_probabilities(X, predicting_batchsize = predicting_batchsize)
		labels = self.context.get_label_names()
		eva_list = []
		for idx, label in enumerate(labels):
			prob = prob_all[:,idx+1]
			y_label = np.array(y == (idx+1))
			eva = self.eva_obj.evaluate(y_label, prob)
			eva_list.append({'label':label, 'evaluation':eva})
		return eva_list


class NextAction_Ensemble_Base(EnsembleModel_Base, NextAction_Base):
	def __init__(self, context, model_config):
		NextAction_Base.__init__(self, context, model_config)
		EnsembleModel_Base.__init__(self, context, model_config)