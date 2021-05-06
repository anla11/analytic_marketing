import numpy as np
import pandas as pd
from propensity_prediction.tasks.abstract import Abstract_Task, Abstract_Model
from model.feature_processing.feature_engineering import Feature_Engineering
import time

def _missingvalue_processing(data_df, context, mode = 'training'):
	if mode == 'training':
		dropna_cols = context.get_feature_names() + context.get_label_names()
		data_df = data_df.dropna(subset=dropna_cols)
	else: 
		fillna_cols = context.get_feature_names() 
		for col in fillna_cols:
			data_df[col].fillna(data_df[col].mean(),inplace=True)
	return data_df

class Model_Base(Abstract_Model, Abstract_Task):
	def __init__(self, context, model_config):
		self.context = context
		self.model_name = model_config['model_name']
		self.fe_obj = None
		self.n_features = None
		self.training_constant = {}

		if 'feature_engineering' in model_config.keys():
			self.fe_obj = Feature_Engineering(model_config['feature_engineering'])
			if len(self.fe_obj.list_config) > 0:		
				model_config['in_dim'] = self.fe_obj.output_dim

		if ('in_dim' in model_config.keys()) and ((model_config['in_dim'] is None) == False):
			self.n_features = model_config['in_dim']
		else:
			self.n_features = len(self.context.get_feature_names())

		self.model, self.eva_obj = None, None #update outside of base classes

	def preprocess(self, data_df):  
		return data_df

	def custom_preprocess_inmodel(self, data_df):  # custom for models
		return data_df

	def _remove_context(self, data_df, mode = 'testing'):
		new_df = self.context.prepare_data(data_df)
		new_df = self.preprocess(new_df)
		new_df = self.custom_preprocess_inmodel(new_df)
		new_df = _missingvalue_processing(new_df, self.context, mode = mode)
		return new_df

	def _get_features(self, noncontext_df):  # return a numpy array of features
		list_features = self.context.get_feature_names()
		data_features = noncontext_df[list_features]
		return np.array(data_features)

	def _get_labels(self, noncontext_df):  # return a numpy array of labels
		labels = self.context.get_label_names()
		y = noncontext_df[labels].values.ravel()
		return np.array(y)

	def _get_ids(self, noncontext_df):
		ids = self.context.get_id_names()
		if ids is None:
			return None
		return noncontext_df[ids]

	def prepare_npdata(self, data_df, get_label=True, mode = 'training'):  
		new_df = self._remove_context(data_df, mode = mode)
		X = self._get_features(new_df)
		X = self.fe_obj.process(X)
		y = None
		if get_label:
			y = self._get_labels(new_df)

		fe_constants = self.get_trainingconstant('feature_engineering')
		X, _, fe_constants = self.model_base.fe_obj.process(X, y, fe_constants)
		self.update_trainingconstant('feature_engineering', fe_constants)
		ids = self._get_ids(new_df)
		return X, y, ids

	def train(self, data_df, training_batchsize = 100): 
		print('Train function %s' % self.model_name)
		X, y, _ = self.prepare_npdata(data_df, mode = 'training')
		self._train(X, y, training_batchsize = training_batchsize)
		return self

	def get_probabilities(self, data_df, predicting_batchsize = 100): 
		X, _, ids = self.prepare_npdata(data_df, get_label=False, mode = 'testing')
		probs = self._get_probabilities(X, predicting_batchsize = predicting_batchsize)
		return {'model_name': self.model_name, 'probabilities': probs, 'id': ids}

	def predict(self, data_df, predicting_batchsize = 100): 
		X, _, ids = self.prepare_npdata(data_df, get_label=False, mode = 'testing')
		return {'model_name': self.model_name, 'predict': self._predict(X, predicting_batchsize = predicting_batchsize), 'id': ids}

	def evaluate(self, data_df, predicting_batchsize = 100): 
		X, y = self.prepare_npdata(data_df, get_label=True, mode = 'training')
		return {'model_name': self.model_name, 'evaluation': self._evaluate(X, y, predicting_batchsize = predicting_batchsize)}

	def package_for_save(self):
		save_package = {'model_name': self.model_name,
						'model_package': self.model.package_for_save(),
						'training_constant': self.training_constant
						}
		return save_package

	def load_from_savepackage(self, load_package):
		self.model.load_from_savepackage(load_package['model_package'])
		self.training_constant = load_package['training_constant']
		return self


class EnsembleModel_Base():  # have list_models, each model is an instance of Model_Base
	def __init__(self, context, model_config):
		self.model_base = Model_Base(context, model_config) # object for only saving config and context
		# self._create_sub_models(model_config, Model_Base, [])

	def _create_sub_models(self, model_config, Model_obj, passing_config = []):
		passing_config.append('in_dim')
		self.list_models = []
		for sub_model_config in model_config['list_models']: 
			sub_model_config.update(self._passing_modelconfig(model_config, passing_config))
			model = Model_obj(self.model_base.context, sub_model_config)
			self.list_models.append(model)

	def _passing_modelconfig(self, model_config, config_names = []):
		return dict(zip(config_names, list([model_config[name] for name in config_names])))

	def prepare_npdata(self, data_df, get_label=True, mode = 'training'):
		data_df = _missingvalue_processing(data_df, self.model_base.context, mode = mode)
		X = self.model_base._get_features(data_df)
		y = None
		if get_label:
			y = self.model_base._get_labels(data_df)

		fe_constants = self.get_trainingconstant('feature_engineering')
		X, _, fe_constants = self.model_base.fe_obj.process(X, y, fe_constants)
		self.update_trainingconstant('feature_engineering', fe_constants)
		ids = self.model_base._get_ids(data_df)
		return X, y, ids

	def train(self, data_df, training_batchsize = 100):
		print ('Prepare contextual data')
		new_df = self.model_base.context.prepare_data(data_df)
		print ('Prepare data for models')
		new_df = self.model_base.preprocess(new_df)

		print('Train ensemble %s' % self.model_name)
		for model in self.list_models:
			start = time.time()

			print('+ Prepare model %s with label %s' % (model.model_name, model.context.get_label_names()[0]))
			custom_df = model.custom_preprocess_inmodel(new_df)
			custom_df = self.model_base.custom_preprocess_inmodel(custom_df)
			X, y, _ = self.prepare_npdata(custom_df, mode = 'training')
			model._train(X, y, training_batchsize = training_batchsize)

			end = time.time()
			# if not(logging is None):
			# 	logging.log('model', (model.model_name, end-start))
			print ('----Running time: ', end-start)

		return self

	def _test(self, data_df, predicting_batchsize = 100, test_func = 'predict', mode = 'testing'):
		new_df = self.model_base.context.prepare_data(data_df)
		new_df = self.model_base.preprocess(new_df)
		res_list = []
		for model in self.list_models:
			custom_df = model.custom_preprocess_inmodel(new_df)
			custom_df = self.model_base.custom_preprocess_inmodel(custom_df)
			get_label = (mode == 'training')
			X, y, ids = self.prepare_npdata(custom_df, get_label = get_label, mode = mode)
			package = None
			if test_func == 'predict':
				res = model._predict(X, predicting_batchsize = predicting_batchsize)
			elif test_func == 'get_probabilities':
				res = model._get_probabilities(X, predicting_batchsize = predicting_batchsize)
			elif test_func == 'evaluate':
				res = model._evaluate(X, y, predicting_batchsize = predicting_batchsize)
			if test_func == 'evaluate':
				package = {'model_name': model.model_name, 'label': list(model.context.get_label_names())[0], 'results': res}
			else:
				package = {'model_name': model.model_name, 'label': list(model.context.get_label_names())[0], 'results': res, 'id': ids}
			res_list.append(package)
		return res_list

	def invalid_process(self, result_arr):
		return result_arr

	def _get_ensemble_results(self, modelres_list, agg_func = None):
		all_model_res = {}
		for model_res in modelres_list:
			res_df = model_res['id']
			res_df.loc[:, model_res['model_name']] = self.invalid_process(np.array(model_res['results']))			
			all_model_res[model_res['model_name']] = res_df

		all_res_df = None
		for model_name, res_df in all_model_res.items():
			if all_res_df is None:
				all_res_df = res_df.copy()
			else:
				all_res_df = all_res_df.merge(res_df, how = 'outer', on = self.model_base.context.get_id_names())

		if (agg_func is None) == False:
			all_res = []
			for model_name in all_model_res.keys():
				all_res.append(all_res_df[model_name])
			if agg_func == 'mean':
				all_res_df.loc[:, 'results'] = pd.DataFrame(all_res).mean(axis = 0)
		return all_res_df

	def _test_ensemble(self, data_df, predicting_batchsize = 100, test_func = 'predict', mode = 'testing'):
		res_list = self._test(data_df, predicting_batchsize = predicting_batchsize, test_func = test_func, mode = mode)
		res_ensemble = self._get_ensemble_results(res_list, agg_func = 'mean')
		return res_ensemble		

	def get_probabilities(self, data_df, predicting_batchsize = 100, mode = 'testing'):
		prob = self._test_ensemble(data_df, predicting_batchsize, 'get_probabilities', mode)
		return {'model_name': self.model_name, 'probabilities': prob}

	def predict(self, data_df, predicting_batchsize = 100, mode = 'testing'):
		pred = self._test_ensemble(data_df, predicting_batchsize, 'predict', mode)
		return {'model_name': self.model_name, 'predict': pred}

	def _get_evalist(self, data_df, predicting_batchsize = 100, mode = 'training'):
		eva_list = self._test(data_df, predicting_batchsize = predicting_batchsize, test_func = 'evaluate', mode = mode)
		# eva_list = self._test_ensemble(data_df, predicting_batchsize, 'evaluate')
		return {'model_name': self.model_name, 'evaluation': eva_list}

	def evaluate(self, data_df, predicting_batchsize = 100, res_func = 'predict'):
		eva_res = self._get_evalist(data_df, predicting_batchsize = predicting_batchsize, mode = 'training')
		res_ensemble = None
		if res_func == 'predict':
			res_ensemble = self.predict(data_df, predicting_batchsize = predicting_batchsize, mode = 'training')['predict']['results']
		elif res_func == 'get_probabilities':
			res_ensemble = self.get_probabilities(data_df, predicting_batchsize = predicting_batchsize, mode = 'training')['probabilities']['results']
		y = self._get_labels(self._remove_context(data_df, mode = 'training'))
		eva_ensemble = self.eva_obj.evaluate(np.array(y), np.array(res_ensemble))
		return {'model_name': self.model_name, 'results': eva_ensemble, 'evaluate_sub_models': eva_res['evaluation']}

	# def _get_list_probabilities(self, data_df, predicting_batchsize = 100):
	# 	new_df = self.model_base.context.prepare_data(data_df)
	# 	new_df = self.model_base.preprocess(new_df)
	# 	prob_list = []
	# 	for model in self.list_models:
	# 		custom_df = model.custom_preprocess_inmodel(new_df)
	# 		custom_df = self.model_base.custom_preprocess_inmodel(custom_df)
	# 		X, _, ids = self.prepare_npdata(custom_df, get_label=False, mode = 'testing')
	# 		prob = model._get_probabilities(X, predicting_batchsize = predicting_batchsize)
	# 		prob_list.append({'model_name': model.model_name, 'label': list(model.context.get_label_names())[0],
	# 						  'probabilities': prob, 'id': ids})
	# 	return prob_list

	# def _get_list_prediction(self, data_df, predicting_batchsize = 100):
	# 	new_df = self.model_base.context.prepare_data(data_df)
	# 	new_df = self.model_base.preprocess(new_df)
	# 	predict_list = []
	# 	for model in self.list_models:
	# 		custom_df = model.custom_preprocess_inmodel(new_df)
	# 		custom_df = self.model_base.custom_preprocess_inmodel(custom_df)
	# 		X, _, ids = self.prepare_npdata(custom_df, get_label=False, mode = 'testing')
	# 		predict = model._predict(X, predicting_batchsize = predicting_batchsize)
	# 		predict_list.append(
	# 			{'model_name': model.model_name, 'label': list(model.context.get_label_names())[0], 'predict': predict, 'id': ids})
	# 	return predict_list

	# def _get_list_evaluation(self, data_df, predicting_batchsize = 100):
	# 	new_df = self.model_base.context.prepare_data(data_df)
	# 	new_df = self.model_base.preprocess(new_df)
	# 	eva_list = []
	# 	for model in self.list_models:
	# 		custom_df = model.custom_preprocess_inmodel(new_df)
	# 		custom_df = self.model_base.custom_preprocess_inmodel(custom_df)
	# 		X, y, _ = self.prepare_npdata(custom_df, mode = 'training')
	# 		eva = model._evaluate(X, y, predicting_batchsize = predicting_batchsize)
	# 		eva_list.append(
	# 			{'model_name': model.model_name, 'label': list(model.context.get_label_names())[0], 'evaluation': eva})
	# 	return {'model_name': self.model_name, 'evaluation': eva_list}

	def package_for_save(self):
		self.update_trainingconstant('context', self.model_base.context.get_constants())
		model_package = []
		for model in self.list_models:
			model_package.append(model.package_for_save())
		return {'model_name': self.model_name, 'model_package': model_package, \
						'training_constant': self.model_base.training_constant}

	def load_from_savepackage(self, load_package):
		load_models = []
		for model, package in zip(self.list_models, load_package['model_package']):
			load_models.append(model.load_from_savepackage(package))
		self.list_models = load_models
		self.model_base.training_constant = load_package['training_constant']
		self.model_base.context.set_constants(load_package['training_constant']['context'])
		return self

	def get_trainingconstant(self, key):
		if key in self.model_base.training_constant.keys():
			return self.model_base.training_constant[key]
		return None

	def update_trainingconstant(self, key, value):
		self.model_base.training_constant[key] = value
		for model in self.list_models:
			model.training_constant[key] = value


class Task_Base(Abstract_Task):
	def __init__(self, context, list_model_config):
		self.model, self.model_name = None, ''
		self.list_models = []  # list of Ensemble Models (instances created from EnsembleModel_Base)

	def train(self, data_df, training_batchsize = 100):
		for model in self.list_models:
			model.train(data_df, training_batchsize = training_batchsize)
		return self.list_models

	def predict(self, data_df, predicting_batchsize = 100):
		predict_list = []
		for model in self.list_models:
			predict_list.append(model.predict(data_df, predicting_batchsize = predicting_batchsize))
		return {'predict': predict_list}

	def get_probabilities(self, data_df, predicting_batchsize = 100):
		prob_list = []
		for model in self.list_models:
			prob_list.append(model.get_probabilities(data_df, predicting_batchsize = predicting_batchsize))
		return {'probabilities': prob_list}

	def evaluate(self, data_df, predicting_batchsize = 100):
		eva_list = []
		for model in self.list_models:
			eva_list.append(model.evaluate(data_df, predicting_batchsize = predicting_batchsize))
		return {'evaluation': eva_list}