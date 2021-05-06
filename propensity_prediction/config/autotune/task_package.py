import sys
import json 
import numpy as np 
import pandas as pd 
from propensity_prediction.auto_preprocess.base import Base_Preprocess
from propensity_prediction.config.base import Base_TaskConfig
from sklearn.model_selection import train_test_split


class Task_Package:
	def __init__(self, global_config, ModelConfig_Class, DataConfig_Class, Task_Class):
		self.global_config = global_config
		self.ModelConfig_Class = ModelConfig_Class
		self.DataConfig_Class = DataConfig_Class
		self.Task_Class = Task_Class

	def _create_taskconfig(self, model_config):
		dataconfig_obj = self.DataConfig_Class(self.global_config['data_config'])
		modelconfig_obj = self.ModelConfig_Class(model_config)
		return Base_TaskConfig(dataconfig_obj, modelconfig_obj)

	def exec(self, model_config):
		task_config = self._create_taskconfig(model_config)

		data_path = task_config.get_datapath()
		key_types, feature_types = task_config.parse_data_config()
		df = pd.read_csv(data_path, dtype=str)

		auto_preprocess_obj = Base_Preprocess(key_types, feature_types)
		data_preprocessed, preprocessed_feature_types = auto_preprocess_obj.auto_preprocess(df)

		data_train, data_val = train_test_split(data_preprocessed, test_size = 0.2, random_state = 0)
		task_config.update_metadata(preprocessed_feature_types)

		ens_model = self.Task_Class(task_config)
		ens_model.train(data_train, training_batchsize = int(len(data_train)/5))

		eva_res = ens_model.evaluate(data_val, predicting_batchsize = int(len(data_val)))
		return eva_res, ens_model

	def get_evaluation(self, model_config):
		raise NotImplementedError() 