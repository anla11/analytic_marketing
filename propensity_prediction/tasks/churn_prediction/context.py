from propensity_prediction.tasks.abstract import Abstract_Context
from model import Survival_Analysis
from model.utils import Batch_Data
import numpy as np

class ChurnPrediction_Context(Abstract_Context):
	def __init__(self, data_config):
		self.sa_obj = Survival_Analysis(data_config) 
		self.input_config = data_config
		self.user_id = data_config.get_column_names(key='user_id')
		self.dur_col = data_config.get_column_names(key='user_seniority')
		self.event_col = data_config.get_column_names(key='event')
		self.metadata_col = data_config.get_column_names(key='metadata')
		self.sur_prob_col = self.sa_obj.sur_prob_col
		self.feature_names = None

	def get_feature_names(self):
		if self.feature_names is None:
			self.feature_names = self.metadata_col + [self.sur_prob_col]
		return self.feature_names

	def get_label_names(self):
		return [self.event_col]

	def get_id_names(self):
		if self.user_id is None:
			return None
		return [self.user_id]

	def set_id_names(self, user_id_name):
		self.user_id = user_id_name

	def __add_survival_features(self, data_df):
		new_df = data_df # data_df.copy()
		if self.sa_obj.kmf is None:
			self.sa_obj._fit_kmf(data_df)
		cumulative_kmf_surprob = self.sa_obj.get_cumulative_kmf_surprob(data_df[self.sa_obj.dur_col])
		new_df.loc[:, self.sur_prob_col] = cumulative_kmf_surprob

		return new_df

	def prepare_data(self, data_df):
		data_df.loc[:, self.user_id] = data_df[self.user_id].apply(lambda r: str(r))
		for key_col in [self.dur_col, self.event_col]:
			data_df.loc[:, key_col] = np.array(data_df[key_col].astype('category').cat.codes).astype(int)
		new_df = self.__add_survival_features(data_df)
		return new_df