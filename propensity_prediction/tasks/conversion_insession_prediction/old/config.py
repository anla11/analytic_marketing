''' This file will be moved to config later '''

import numpy as np
import pandas as pd

def parse_data_config(global_config):
	data_path = global_config['data_config']['path']
	feature_types = global_config['data_config']['feature_types']
	return data_path, feature_types

def generate_pipeline_config(global_config, preprocess_features):
	INPUT_CONFIG, FE_CONFIG, MODEL_CONFIG, PREDICT_CONFIG = None, None, None, None
	if global_config['task'] == 'converting_action_prediction':
		data_config = global_config['data_config']
		INPUT_CONFIG = {
			'user_id':data_config['cus_id'],
			'product_id': data_config['product_id'],
			'user_session':data_config['session'],
			'event': data_config['event'],
			'order_actions': data_config['order_actions'],
			'other_features': preprocess_features
		}
		FE_CONFIG = []
		MODEL_CONFIG=['BinaryClasses','MultiClass']
		PREDICT_CONFIG={'method':'gettop','ntop':None}
	PIPELINE_CONFIG = {'task': global_config['task'], 'input_config':INPUT_CONFIG, 'fe_config':FE_CONFIG, 'model_config':MODEL_CONFIG, 'predict_config':PREDICT_CONFIG}
	return PIPELINE_CONFIG




