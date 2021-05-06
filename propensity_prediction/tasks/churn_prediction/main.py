'''
	@author: anla-ds
	created date: 29 July, 2020
'''
import sys
import json 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

from propensity_prediction.auto_preprocess.base import Base_Preprocess
from propensity_prediction.config.model_config import ChurnPrediction_ModelConfig
from propensity_prediction.config.data_config import ChurnPrediction_DataConfig
from propensity_prediction.config.base import Base_TaskConfig
from propensity_prediction.tasks.churn_prediction.churn_prediction import ChurnPrediction_Task

class ChurnPrediction_Config(Base_TaskConfig):
	def __init__(self, global_config):
		data_config = ChurnPrediction_DataConfig(global_config['data_config'])
		model_config = ChurnPrediction_ModelConfig(global_config['pipeline_config_path'])
		super().__init__(data_config, model_config)

def churn_prediction(global_config):
	task_config = ChurnPrediction_Config(global_config)

	data_path = task_config.get_datapath()
	key_types, feature_types = task_config.parse_data_config()
	df = pd.read_csv(data_path, dtype=str)

	auto_preprocess_obj = Base_Preprocess(key_types, feature_types)
	data_preprocessed, preprocessed_feature_types = auto_preprocess_obj.auto_preprocess(df)

	data_train, data_val = train_test_split(data_preprocessed, test_size = 0.2, random_state = 0)
	task_config.update_metadata(preprocessed_feature_types)

	ens_model = ChurnPrediction_Task(task_config)
	ens_model.train(data_train, training_batchsize = 1000)

	eva_res = ens_model.evaluate(data_val, predicting_batchsize = 1000)
	predict = ens_model.predict(data_preprocessed, predicting_batchsize = 1000)

	return predict, eva_res, ens_model

if __name__=='main':
	json_file = open(sys.argv[1], r) 
	global_config = json.load(json_file)
	churn_prediction(global_config)