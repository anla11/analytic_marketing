from propensity_prediction.config.datagroup import USER_PROFILE, PRODUCT_PROFILE, HISTORY, CONSTRAINT
from propensity_prediction.config.base import Base_TaskConfig, Base_DataConfig, HistoryBased_DataConfig, SessionBased_DataConfig

class ChurnPrediction_DataConfig(HistoryBased_DataConfig):
	def __init__(self, data_config):
		super().__init__(data_config)

class LTVPrediction_DataConfig(HistoryBased_DataConfig):
	def __init__(self, data_config):
		super().__init__(data_config)		

class Conversion_Insession_Prediction_DataConfig(SessionBased_DataConfig):
	def __init__(self, data_config):
		super().__init__(data_config)
		
# class ConvertingActionPrediction_DataConfig(Base_DataConfig):
# 	def __init__(self, data_config):
# 		user_profile, product_profile=None, None
# 		history = HISTORY()
# 		constraint = CONSTRAINT()
# 		history.update_config(data_config['History'])
# 		constraint.update_config(data_config['Constraint'])
# 		super().__init__(user_profile, product_profile, history, constraint)
# 		if 'path' in  data_config.keys():
# 			self.data_path = data_config['path']
	
# 	def get_column_names(self, key):
# 		if key in self.history.get_attributes():
# 			return self.history.get_column(key)
# 		elif key in self.constraint.get_attributes():
# 			return self.constraint.get_column(key)

# 	def parse_data_config(self):
# 		key_types = self.get_key_preprocess_types()
# 		feature_types = self.history.get_datafeatures()
# 		return key_types, feature_types

# 	def get_key_preprocess_types(self):
# 		list_keys = self.history.get_datakeys()
# 		list_keys = list(filter(None, list_keys)) 
# 		key_preprocess_types = [None]*len(list_keys)
# 		keys_types = dict(zip(list_keys, key_preprocess_types))
# 		return keys_types

# 	def update_metadata(self, new_metadata):			
# 		self.history.metadata = new_metadata


