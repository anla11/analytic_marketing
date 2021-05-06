''' This file will be moved to config later '''
import numpy as np
import pandas as pd

def parse_data_config(global_config):
	data_path = global_config['data_config']['path']
	feature_types = global_config['data_config']['feature_types']
	return data_path, feature_types

def generate_pipeline_config(global_config, preprocess_features):
	INPUT_CONFIG, FE_CONFIG, MODEL_CONFIG, PREDICT_CONFIG = None, None, None, None
	if global_config['task'] == 'LTV_prediction':
		data_config = global_config['data_config']
		INPUT_CONFIG = {
			'cus_id': data_config['cus_id'],
			'invoice_date': data_config['invoice_date'],
			'invoice_no':data_config['invoice_no'],
			'num_units': data_config['num_units'],
			'unit_price': data_config['unit_price'],
			'months_train': data_config['months_train']
		}
		FE_CONFIG = []
		MODEL_CONFIG=['Regression']
		PREDICT_CONFIG={'method':'gettop','ntop':None}
	PIPELINE_CONFIG = {'task': global_config['task'], 'input_config':INPUT_CONFIG, 'fe_config':FE_CONFIG, 'model_config':MODEL_CONFIG, 'predict_config':PREDICT_CONFIG}
	return PIPELINE_CONFIG




