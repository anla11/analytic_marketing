import sys
import json 

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


from propensity_prediction.auto_preprocess.base import Base_Preprocess

from propensity_prediction.config.model_config import LTVPrediction_ModelConfig
from propensity_prediction.config.data_config import LTVPrediction_DataConfig
from propensity_prediction.config.base import Base_TaskConfig
from propensity_prediction.tasks.LTV_prediction.LTV_prediction import LTVPrediction_Task


class LTVPrediction_Config(Base_TaskConfig):
	def __init__(self, global_config):
		data_config = LTVPrediction_DataConfig(global_config['data_config'])
		model_config = LTVPrediction_ModelConfig()
		super().__init__(data_config, model_config)

def LTV_prediction(global_config):
	task_config = LTVPrediction_Config(global_config)
	
	data_path = task_config.get_datapath()
	key_types, feature_types = task_config.parse_data_config()
	df = pd.read_excel(data_path)

	auto_preprocess_obj = Base_Preprocess(key_types, feature_types)
	data_df, preprocess_feature_types = auto_preprocess_obj.auto_preprocess(df)

	# date_column_name = task_config.data_config.get_column_name(key='invoice_date') 
	# data_train, data_test = data_df[data_df[date_column_name]<''], data_df[data_df[date_column_name]>=''] 

	data_train, data_test = train_test_split(data_df, test_size = 0.2, random_state = 0)
	task_config.update_metadata(preprocess_feature_types)

	ens_model = LTVPrediction_Task(task_config)
	ens_model.train(data_test, training_batchsize = 1000)

	predict = ens_model.predict(data_df, predicting_batchsize = 1000)
	evaluate = ens_model.evaluate(data_test, predicting_batchsize = 1000)
	return predict, evaluate, ens_model

if __name__=='main':
	json_file = open(sys.argv[1], r) 
	global_config = json.load(json_file)
	ltv_prediction(global_config)




