# objects saving columns'names of data

class Data_Group():
	def generate_config(self):
		return self.__dict__

	def update(self, key, value): # update attributes to the object
		setattr(self, key, value)
	
	def update_config(self, config):
		for k, v in config.items():
			setattr(self, k, v)
	
	def get_attributes(self):     #get all atributes 
		config = self.__dict__
		return list(config.keys())

	def get_columns(self, key):   #key: attributes, return: column names in data_df
		config = self.__dict__
		if key != 'metadata':
			return config[key]
		else:
			return list(config[key].keys())

	def get_key_names(self):      #get names of all id fields (not metadata fields) 
		list_datakeys = []
		config = self.__dict__
		for field, col in config.items():
			if field != 'metadata':
				list_datakeys.append(col)
		return list_datakeys

	def get_keys(self):           #get names and types of all id fields (not metadata fields) 
		list_keys = self.get_key_names()
		list_keys = list(filter(None, list_keys)) 		
		types = ['key']*len(list_keys)
		keys_types = dict(zip(list_keys, types))
		return keys_types

	def get_features(self):       #features = values of metadata features of data_df
		return self.metadata

	def get_feature_names(self):  #get names of features
		return list(self.metadata.keys())

	def get_feature_types(self):  #get types of features
		return list(self.metadata.values())

class USER_PROFILE(Data_Group):
	def __init__(self):
		self.user_id = 'user_id'
		self.metadata = {}

class PRODUCT_PROFILE(Data_Group):
	def __init__(self):
		self.product_id = 'product_id'
		self.metadata = {}

class HISTORY(Data_Group):
	def __init__(self):
		self.user_id = 'user_id'
		self.metadata = {}	

class SESSION(Data_Group):
	def __init__(self):
		self.user_id = 'user_id'
		self.metadata = {}	

class CONSTRAINT(Data_Group):
	def __init__(self):
		self.constraint_field = 'constraint_field'
		self.constraint_value = 'constraint_value'
