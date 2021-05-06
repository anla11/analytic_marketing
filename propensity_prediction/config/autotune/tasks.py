import numpy as np
from propensity_prediction.config.autotune.task_package import Task_Package
from propensity_prediction.config.model_config import Base_ModelConfig


''' Churn Prediction '''
from propensity_prediction.config.data_config import ChurnPrediction_DataConfig
from propensity_prediction.tasks.churn_prediction.churn_prediction import ChurnPrediction_Task

class Tune_ChurnPrediction_ModelConfig(Base_ModelConfig):
	def __init__(self, pipeline_config):
		super().__init__('Ensemble', default_config = pipeline_config, config_path = None, list_configs = ['binarize_config', 'feature_engineering', 'model_config'])

class ChurnPrediction_Package(Task_Package):
	def __init__(self, global_config):
		super().__init__(global_config, Tune_ChurnPrediction_ModelConfig, ChurnPrediction_DataConfig, ChurnPrediction_Task)

	def get_evaluation(self, model_config):  #lower is better
		eva_res, ens_model = self.exec(model_config)
		bin_method = model_config['binarize_config']['method']
		res, order = -np.inf, -1 #order == -1 means higher is better

		if bin_method == 'gettop':
			res = eva_res['results']['binarize_methods'][0]['list_results'][1]['results']['f1_score']
		elif bin_method == 'threshold':
			threshold_results = eva_res['results']['binarize_methods'][0]['list_results']
			for i in range(len(threshold_results)):
				res_obj = threshold_results[i]
				thres_method, f1_score = res_obj['threshold_method'], res_obj['results']['f1_score']
				if thres_method == model_config['binarize_config']['para']['threshold_method']:
					res = f1_score

		return res, order

''' Conversion In Session'''
from propensity_prediction.config.data_config import Conversion_Insession_Prediction_DataConfig
from propensity_prediction.tasks.conversion_insession_prediction.conversion_insession_prediction import Conversion_InSession_Task


class Tune_ConversionInSession_Prediction_ModelConfig(Base_ModelConfig):
	def __init__(self, pipeline_config):
		super().__init__('Ensemble', default_config = pipeline_config, config_path = None, list_configs = ['binarize_config', 'feature_engineering', 'model_config'])

class ConversionInSession_Prediction_Package(Task_Package):
	def __init__(self, global_config):
		super().__init__(global_config, Tune_ConversionInSession_Prediction_ModelConfig, Conversion_Insession_Prediction_DataConfig, Conversion_InSession_Task)

	def get_evaluation(self, model_config):  #lower is better
		try:    
			eva_res, ens_model = self.exec(model_config)
			bin_method = model_config['binarize_config']['method']
			res, order = -np.inf, -1 #order == -1 means higher is better

			if bin_method == 'gettop':
				res = eva_res['results']['binarize_methods'][0]['list_results'][1]['results']['f1_score']
			elif bin_method == 'threshold':
				threshold_results = eva_res['results']['binarize_methods'][0]['list_results']
				for i in range(len(threshold_results)):
					res_obj = threshold_results[i]
					thres_method, f1_score = res_obj['threshold_method'], res_obj['results']['f1_score']
					if thres_method == model_config['binarize_config']['para']['threshold_method']:
						res = f1_score

			return res, order
		except: 
			return -np.inf, -1


''' LTV prediction '''
from propensity_prediction.config.data_config import LTVPrediction_DataConfig
from propensity_prediction.tasks.LTV_prediction.LTV_prediction import LTVPrediction_Task


class Tune_LTV_Prediction_ModelConfig(Base_ModelConfig):
	def __init__(self, pipeline_config):
		super().__init__('Ensemble', default_config = pipeline_config, config_path = None, list_configs = ['feature_engineering', 'model_config'])

class LTV_Prediction_Package(Task_Package):
	def __init__(self, global_config):
		super().__init__(global_config, Tune_LTV_Prediction_ModelConfig, LTVPrediction_DataConfig, LTVPrediction_Task)

	def get_evaluation(self, model_config):  #lower is better
		eva_res, ens_model = self.exec(model_config)
		res, order = -np.inf, -1 #order == -1 means higher is better
		res = eva_res['results']['r2_score']
		return res, order


def get_task_package(global_config):
	task_name = global_config['task_name']
	task_package = None
	if task_name == 'churn_prediction':
		task_package = ChurnPrediction_Package(global_config)
	if task_name == 'conversion_insession_prediction':
		task_package = ConversionInSession_Prediction_Package(global_config)
	if task_name == 'ltv_prediction':
		task_package = LTV_Prediction_Package(global_config)
	return task_package

def get_pipeline_space(global_config):
	task_name = global_config['task_name']
	if task_name == 'churn_prediction':
		from propensity_prediction.config.autotune.param_space import classification_pipeline_space
		return classification_pipeline_space
	if task_name == 'conversion_insession_prediction':
		from propensity_prediction.config.autotune.param_space import classification_pipeline_space
		return classification_pipeline_space
	if task_name == 'ltv_prediction':
		from propensity_prediction.config.autotune.param_space import regression_pipeline_space
		return regression_pipeline_space
	return None


