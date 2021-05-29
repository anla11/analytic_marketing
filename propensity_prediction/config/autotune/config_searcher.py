# from propensity_prediction.config.autotune.param_space import classification_pipeline_space
import numpy as np
import optuna
from model.utils.json_processing import save_json, load_json

def get_values(x = {}, key = ''):
	if x is None:
		return None
	if key in x.keys():
		return x[key]
	return None 

def find_para(trial, space_name, suggest_name, search_space):
	value = None
	if suggest_name == 'choice':
		value = trial.suggest_categorical(space_name, search_space)
	if suggest_name == 'int':
		value = trial.suggest_int(space_name, low = search_space['low'], high = search_space['high'], log = search_space['log'], step = search_space['step']) 	
	if suggest_name == 'float':
		value = trial.suggest_float(space_name, low = search_space['low'], high = search_space['high'], log = search_space['log'], step = search_space['step'])	
	if suggest_name == 'uniform':			
		value = trial.suggest_uniform(space_name, search_space, low = search_space['low'], high = search_space['high'])
	if suggest_name == 'loguniform':
		value = trial.suggest_loguniform(space_name, search_space, low = search_space['low'], high = search_space['high'])
	if suggest_name == 'discrete_uniform':
		value = trial.suggest_discrete_uniform(space_name, low = search_space['low'], high = search_space['high'], q = search_space['q'])
	return value

class ConfigUnit:
	def __init__(self, config):
		self.para_name, self.space_name = get_values(config,'para_name'), get_values(config,'space_name')
		self.space, self.sub_space = get_values(config,'space'), get_values(config,'sub_space')
		self.suggest, self.child_relationship = get_values(config,'suggest'), get_values(config,'child_relationship')  
				#leaf or 'joint' or 'conditional' or 'multi-para'

	def get_define(self):
		return self.para_name, self.space_name, self.suggest, self.space      

class ConfigTuner:
	def get_para_trial(self, trial, config_node):
		para = None
		if ((config_node.space is None) == False) and (len(config_node.space) > 0):
			para = find_para(trial, config_node.space_name, config_node.suggest, config_node.space)
		return para

	def _create_leaf_config(self, trial, config_node):
		if config_node.para_name is None: 
			return {}
		para = self.get_para_trial(trial, config_node)
		return {config_node.para_name: para}

	def _create_joint_config(self, trial, config_node):
		config = {'method': [], 'para': {}}
		for sub_name in config_node.space:
			sub_node = ConfigUnit(config_node.sub_space[sub_name])
			config['method'].append(sub_name)
			config['para'].update({sub_name: self.create_config(trial, sub_node)})
		return config

	def _create_multipara_config(self, trial, config_node):
		config = {}
		for sub_name in config_node.space:
			sub_node = ConfigUnit(config_node.sub_space[sub_name])
			config.update(self.create_config(trial, sub_node))    
		return config

	def _create_conditional_config(self, trial, config_node):
		config = {}
		para = self.get_para_trial(trial, config_node)
		sub_node = ConfigUnit(config_node.sub_space[para])
		config.update({'method': para})
		config.update({'para': {para: self.create_config(trial, sub_node)}})
		return config 

	def create_config(self, trial, config_node):
		if config_node.child_relationship is None:
			return self._create_leaf_config(trial, config_node)
		if config_node.child_relationship == 'multi-para':
			return self._create_multipara_config(trial, config_node)
		elif config_node.child_relationship == 'joint':
			return self._create_joint_config(trial, config_node)
		elif config_node.child_relationship == 'conditional':
			return self._create_conditional_config(trial, config_node)
		return {}

class ConfigParser:
	def __init__(self, running_config = {'epochs': 500}):
		self.running_config = running_config

	def _parse(self, config, key):    
		value = config[key]
		if len(value.keys())== 0 or list(value.keys())[0] != 'method':
			return value
			
		if type(value['method']) != list:
			return {key: {'method': value['method'], 'para': self._parse(value['para'], value['method'])}}
		elif type(value['method']) == list:
			config = {key: []}
			for method in value['method']:
				sub_config = {'method': method, 'para': value['para'][method]}
				if (key == 'model_config'):
					sub_config['para'].update(self.running_config)
				config[key].append(sub_config)
			return config
		return None
	
	def parse(self, best_pipeline_config):
		key_list = best_pipeline_config['method']
		config = best_pipeline_config['para']   
		pipeline_config = {}
		for key in key_list:
			pipeline_config.update(self._parse(config, key))
		return pipeline_config


class ConfigSearcher:
	def __init__(self, task_package, pipeline_config, running_config = {'epochs': 500}):
		self.list_results = []
		self.best_result, self.best_config = None, None #lower is better
		self.tuner = ConfigTuner()
		self.parser = ConfigParser(running_config)
		self.task_package = task_package
		self.study = optuna.create_study()  # Create a new study.
		self.pipeline_config = pipeline_config

	def objective(self, trial):
		config_obj = self.tuner.create_config(trial, ConfigUnit(self.pipeline_config))
		pipeline_config = self.parser.parse(config_obj)

		res, order = None, None
		try:
			res, order = self.task_package.get_evaluation(pipeline_config)
		except:
			print ("Error!")
		self.list_results.append(res)

		if (res is None) or np.isnan(res) or np.isinf(abs(res)):
			return np.inf

		if ((self.best_result is None) or (order * res < order * self.best_result)):
			self.best_result = res
			self.best_config = pipeline_config

		return res * order

	def search(self, n_trials = 50):
		self.study.optimize(self.objective, n_trials=n_trials)  # Invoke optimization of the objective function.
	  
	def save_bestconfig(self, path):
		save_json(self.best_config, path)

	def load_bestconfig(self, path):
		return load_json(path)       