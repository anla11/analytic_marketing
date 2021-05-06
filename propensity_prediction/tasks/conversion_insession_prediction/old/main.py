import sys
import json 
import pandas as pd
import numpy as np

# from propensity_prediction.utils.transform import scale_minmax
from sklearn.model_selection import train_test_split

# from propensity_prediction.model.feature_processing.preprocessing import auto_preprocess
from propensity_prediction.auto_preprocess.base import Base_Preprocess

from propensity_prediction.config.model_config import ConvertingActionPrediction_ModelConfig
from propensity_prediction.config.data_config import ConvertingActionPrediction_DataConfig
from propensity_prediction.config.base import Base_TaskConfig
from propensity_prediction.tasks.converting_action_prediction.converting_action_prediction import NextAction_InSession_Task


class ConvertingActionPrediction_Config(Base_TaskConfig):
	def __init__(self, global_config):
		data_config = ConvertingActionPrediction_DataConfig(global_config['data_config'])
		model_config = ConvertingActionPrediction_ModelConfig()
		super().__init__(data_config, model_config)

def converting_action_prediction(global_config):
	
	task_config = ConvertingActionPrediction_Config(global_config)
	data_path = task_config.get_datapath()
	key_types, feature_types = task_config.parse_data_config()

	df = pd.read_csv(data_path, dtype=str)
	
	# user_col = global_config['data_config']['cus_id']
	# product_col = global_config['data_config']['product_id']
	# event_col = global_config['data_config']['event']
	# session_col = global_config['data_config']['session']
	# data_df, preprocess_feature_types = auto_preprocess(df, key_types, feature_types)
	auto_preprocess_obj = Base_Preprocess(key_types, feature_types)
	data_df, preprocess_feature_types = auto_preprocess_obj.auto_preprocess(df)


	data_train, data_test = train_test_split(data_df, test_size = 0.2, random_state = 0)
	task_config.update_metadata(preprocess_feature_types)

	ens_model = NextAction_InSession_Task(task_config)
	ens_model.train(data_train, training_batchsize = 50000)

	predict = ens_model.predict(data_test, predicting_batchsize = 50000)
	evaluate = ens_model.evaluate(data_test, predicting_batchsize = 50000)
	return predict, evaluate, ens_model

if __name__=='main':
	json_file = open(sys.argv[1], r) 
	global_config = json.load(json_file)
	converting_action_prediction(global_config)
