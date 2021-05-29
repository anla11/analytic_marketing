import numpy as np
import pandas as pd
import os.path

from propensity_prediction.config.abstract import Abstract_ModelConfig, Abstract_DataConfig, Abstract_TaskConfig
from propensity_prediction.config.datagroup import USER_PROFILE, PRODUCT_PROFILE, HISTORY, SESSION, CONSTRAINT

class Base_TaskConfig(Abstract_TaskConfig):
	def __init__(self, data_config = None, model_config = None):
		super().__init__(data_config, model_config)

	def get_datapath(self):
		return self.data_config.get_datapath()
	def parse_data_config(self):
		return self.data_config.parse_data_config()
	def update_metadata(self, new_metadata):
		self.data_config.update_metadata(new_metadata)	


class Base_ModelConfig(Abstract_ModelConfig):
	def get_model_config(self):
		return self.__dict__

	def _update(self, key, value):
		setattr(self, key, value)
	
	def _update_config(self, config):
		for k, v in config.items():
			setattr(self, k, v)

	def _convert_modelconfig(self, model_config):
		new_model_config = {'list_models': []} 
		for model in model_config:
			new_model = {'model_name': model['method'], 'training_config': model['para']}
			new_model_config['list_models'].append(new_model)
		return new_model_config	

	def __init__(self, name, default_config, config_path = None, list_configs = ['feature_engineering', 'model_config']):
		config_raw = None
		if ((config_path is None) == False) and os.path.isfile(config_path):
			from model.utils.json_processing import load_json
			config_raw = load_json(config_path)
		else:
			config_raw = default_config

		pipeline_config = {}
		for config_name in list_configs:
			if config_name != 'model_config':
				pipeline_config[config_name] = config_raw[config_name]
			else:
				pipeline_config.update(self._convert_modelconfig(config_raw['model_config']))

		self.model_name = name
		self._update_config(pipeline_config)
		self.in_dim, self.out_dim = None, None	


class Base_DataConfig(Abstract_DataConfig):
	def __init__(self, user_profile=None, product_profile=None, history=None, session=None, constraint_config=None):
		self.user_profile = user_profile
		self.product_profile = product_profile
		self.history = history
		self.session = session
		self.data_path = None #update later

		if (constraint_config is None) == False:
			self.constraint = CONSTRAINT()
			self.constraint.update_config(constraint_config)
			self.constraint.constraint_key = self.get_column_names(self.constraint.constraint_field)		

	def parse_data_config(self):
		keys_types, feature_types = {}, {}
		for obj in [self.user_profile, self.product_profile, self.history, self.session]:
			if (obj is None) == False:
				keys_types.update(obj.get_keys())
				feature_types.update(obj.get_features())
		return keys_types, feature_types

	def get_column_names(self, key):
		for obj in [self.user_profile, self.product_profile, self.history, self.session]:
			if (obj is None) == False:
				if key in obj.get_attributes():
					return obj.get_columns(key)	

	def get_keys(self):
		keys_types = {}
		for obj in [self.user_profile, self.product_profile, self.history, self.session]:
			if (obj is None) == False:
				keys_types.update(obj.get_keys())
		return keys_types		

	def get_datapath(self):
		return self.data_path

	def get_constraint(self):
		return self.constraint


class HistoryBased_DataConfig(Base_DataConfig):	
	def __init__(self, data_config):
		user_profile, product_profile, session, constraint_config = None, None, None, None
		history = HISTORY()
		history.update_config(data_config['History'])
		if 'Constraint' in data_config.keys():
			constraint_config = data_config['Constraint']
			
		super().__init__(user_profile, product_profile, history, session, constraint_config)
		if 'path' in  data_config.keys():
			self.data_path = data_config['path']

	def update_metadata(self, new_metadata):
		self.history.metadata = new_metadata


class SessionBased_DataConfig(Base_DataConfig):	
	def __init__(self, data_config):
		user_profile, product_profile, history, constraint_config = None, None, None, None
		session = SESSION()
		session.update_config(data_config['Session'])
		if 'Constraint' in data_config.keys():
			constraint_config = data_config['Constraint']

		super().__init__(user_profile, product_profile, history, session, constraint_config)
		if 'path' in  data_config.keys():
			self.data_path = data_config['path']

	def update_metadata(self, new_metadata):			
		self.session.metadata = new_metadata

