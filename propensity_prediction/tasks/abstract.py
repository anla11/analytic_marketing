class Abstract_Context():
	'''
		Prepare data for models, includes
			+ prepare_data: return dataframe preprocessed
			+ get_feature_names, get_label_names: return columns'name in dataframe
		for example: ChurnPrediction_Context in churn_prediction/context.py
	'''
	def get_feature_names(self):
		raise NotImplementedError()	

	def get_label_names(self):
		raise NotImplementedError()

	def get_id_names(self):
		raise NotImplementedError()		

	def prepare_data(self, data_df):
		raise NotImplementedError()

	def get_constants(self):
		return None

	def set_constants(self, constants = None):
		pass 

class Abstract_Model():
	def __init__(self, context, model_config):
		'''
			self.context is object inheriting from Abstract_Context
			self.model is object inheriting from classes of abstract_training.py
				in model/classification or other folders
		'''
		self.context = context
		self.model = None 
		self.model_name = model_config['model_name']

	def preprocess(self, data_df):  #should consider two cases: train|test mode
		raise NotImplementedError() 

	def custom_preprocess_inmodel(self, data_df): #custom for models in ensemble
		raise NotImplementedError() 			
        
	def prepare_npdata(self, data_df, get_label=True, mode = 'training'):  
		raise NotImplementedError()

	def _train(self, X, y, training_batchsize = 100):
		raise NotImplementedError()
        
	def _predict(self, X, predicting_batchsize = 100):
		raise NotImplementedError()        
        
	def _get_probabilities(self, X, predicting_batchsize = 100):
		raise NotImplementedError()

	def _evaluate(self, X, y, predicting_batchsize = 100):
		raise NotImplementedError()  

	# def _get_features(self, data_df): #return a numpy array of features
	# 	pass		
    
	# def _get_labels(self, data_df): #return a numpy array of labels
	# 	pass				

	# def _get_ids(self):
	# 	pass				

	# def package_for_save(self, input_define):
	# 	pass		

	# def load_from_savepackage(self):
	# 	pass  		

class Abstract_Task():
	def prepare_npdata(self, data_df, get_label=True, mode = 'training'):
		raise NotImplementedError()

	def post_process(self, result_arr): #post processing results of models (remove inf/nan values...)
		pass

	def train(self, data_df, training_batchsize = 100):
		raise NotImplementedError()

	def get_probabilities(self, data_df, predicting_batchsize = 100):
		raise NotImplementedError()

	def predict(self, data_df, predicting_batchsize = 100):
		raise NotImplementedError()

	def evaluate(self, data_df, predicting_batchsize = 100):
		raise NotImplementedError()

	# def package_for_save(self, data_df, predicting_batchsize = 100):
	# 	raise NotImplementedError()		

	# def load_from_savepackage(self, data_df, predicting_batchsize = 100):
	# 	raise NotImplementedError()		