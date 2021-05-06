import sys
import json 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

from propensity_prediction.auto_preprocess.base import Base_Preprocess

from propensity_prediction.config.model_config import Conversion_InSession_Prediction_ModelConfig
from propensity_prediction.config.data_config import Conversion_Insession_Prediction_DataConfig
from propensity_prediction.config.base import Base_TaskConfig
from propensity_prediction.tasks.conversion_insession_prediction.conversion_insession_prediction import Conversion_InSession_Task

class Conversion_Insession_Prediction_Config(Base_TaskConfig):
	def __init__(self, global_config):
		data_config = Conversion_Insession_Prediction_DataConfig(global_config['data_config'])
		model_config = Conversion_InSession_Prediction_ModelConfig(global_config['pipeline_config_path'])
		super().__init__(data_config, model_config)

def conversion_insession_prediction(global_config):
	task_config = Conversion_Insession_Prediction_Config(global_config)

	data_path = task_config.get_datapath()
	key_types, feature_types = task_config.parse_data_config()
	df = pd.read_csv(data_path, dtype=str, nrows = 20000)

	auto_preprocess_obj = Base_Preprocess(key_types, feature_types)
	data_preprocessed, preprocessed_feature_types = auto_preprocess_obj.auto_preprocess(df)

	data_train, data_val = train_test_split(data_preprocessed, test_size = 0.2, random_state = 0)
	task_config.update_metadata(preprocessed_feature_types)

	ens_model = Conversion_InSession_Task(task_config)
	ens_model.train(data_train, training_batchsize = 10000)

	eva_res = ens_model.evaluate(data_val, predicting_batchsize = 10000)
	predict = ens_model.predict(data_val, predicting_batchsize = 10000)

	return predict, eva_res, ens_model

if __name__=='main':
	json_file = open(sys.argv[1], r) 
	global_config = json.load(json_file)
	churn_prediction(global_config)