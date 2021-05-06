import pandas as pd 
import numpy as np
from propensity_prediction.tasks.abstract import Abstract_Context
from model import InSessionFeatures_ByCus, InSessionFeatures_ByProd, InSessionFeatures_ByTransaction

class Conversion_InSession_Context(Abstract_Context):
	def __init__(self, data_config):
		self.label_field = data_config.constraint.constraint_key
		self.label_name = data_config.constraint.constraint_value
		self.feature_user = InSessionFeatures_ByCus(data_config)
		self.feature_product = InSessionFeatures_ByProd(data_config)
		self.feature_session = InSessionFeatures_ByTransaction(data_config)

	def get_feature_names(self):
		return self.feature_user.feature_cols + self.feature_product.feature_cols + self.feature_session.feature_cols

	def get_label_names(self):
		return [self.label_name] 

	def get_id_names(self):
		return self.feature_session.keylist
	
	def _get_label_df(self, df):
		df[self.label_name] = np.where(df[self.label_field] == self.label_name, 1, 0)
		label_df = df.groupby(self.feature_session.keylist)[self.label_name].max().reset_index()
		return label_df 

	def prepare_data(self, df):
		# Generate data
		df_preprocessed = self.feature_session.preprocess(df)
		user_insession_profile = self.feature_user.get_profile(df_preprocessed)
		prod_insession_profile = self.feature_product.get_profile(df_preprocessed)
		insession_profile = self.feature_session.get_profile(df_preprocessed)
		label_df = self._get_label_df(df_preprocessed)
		# Merge data
		data_training = insession_profile.merge(user_insession_profile, on = self.feature_session.user_id, how = 'left')
		data_training = data_training.merge(prod_insession_profile, on = self.feature_session.product_id, how = 'left')
		data_training = data_training.merge(label_df, on= self.feature_session.keylist, how = 'left')
		return data_training