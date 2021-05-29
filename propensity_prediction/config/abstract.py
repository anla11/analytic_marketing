class Abstract_ModelConfig():
	def __init__(self):
		self.model_name = None
		self.list_models = None
		self.feature_engineering = None
		self.in_dim, self.out_dim = None, None		
		
class Abstract_DataConfig():
	def __init__(self):
		self	
	def parse_data_config(self):
		raise NotImplementedError()  
	def get_datapath(self):
		raise NotImplementedError()
	def get_column_names(self, key):
		raise NotImplementedError()  
	def get_keys(self):
		raise NotImplementedError()  
	def update_metadata(self, new_metadata):			
		raise NotImplementedError()  

class Abstract_TaskConfig():
	def __init__(self, data_config = None, model_config = None):
		self.data_config = data_config 
		self.model_config = model_config 
