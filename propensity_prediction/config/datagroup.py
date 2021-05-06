# objects saving columns'names of data

class Data_Group():
	def generate_config(self):
		return self.__dict__

	def update(self, key, value):
		setattr(self, key, value)
	
	def update_config(self, config):
		for k, v in config.items():
			setattr(self, k, v)
	
	def get_attributes(self): #get all atributes of datagroup
		config = self.__dict__
		return list(config.keys())

	def get_columns(self, key): #key: attributes of datagroup, return: column names in data_df
		config = self.__dict__
		if key != 'metadata':
			return config[key]
		else:
			return list(config[key].keys())

	def get_key_names(self): 
		list_datakeys = []
		config = self.__dict__
		for field, col in config.items():
			if field != 'metadata':
				list_datakeys.append(col)
		return list_datakeys

	def get_keys(self):
		list_keys = self.get_key_names()
		list_keys = list(filter(None, list_keys)) 		
		types = ['key']*len(list_keys)
		keys_types = dict(zip(list_keys, types))
		return keys_types

	def get_features(self): # features = values of other features of data_df
		return self.metadata

	def get_feature_names(self): 
		return list(self.metadata.keys())

	def get_feature_types(self):
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
		# self.event = 'event' #optional

class SESSION(Data_Group):
	def __init__(self):
		self.user_id = 'user_id'
		self.metadata = {}	
		# self.user_seniority='user_seniority' #optional
		# self.user_session = 'user_session' #optional
		# self.product_id = 'product_id' #optional
		# self.event = 'event' #optional

class CONSTRAINT(Data_Group):
	def __init__(self):
		# self.constraint = {}
		pass

