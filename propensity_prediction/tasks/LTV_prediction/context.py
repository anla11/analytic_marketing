import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import datetime as dt
import numpy as np

from model import compute_clusters, cal_LTV, PurchaseFeatures_ByAge
from model.utils import _generate_sortedcluster

from propensity_prediction.tasks.abstract import Abstract_Context

class LTVPrediction_Context(Abstract_Context):
	def __init__(self, data_config):
		self.LTV = 'LTV'
		self.group_ltv = 'group_ltv'
		self.agerange = 30 # (days)
		self.num_clusters = 3
		self.feature_generator = PurchaseFeatures_ByAge(data_config, activedate_mapping = None, rangelen = self.agerange)
		self.ltvrate_byage = None

	def get_feature_names(self):
		return [self.feature_generator.Customer_Age, 'LTV_rate', self.group_ltv] +  self.feature_generator.featurelist

	def get_label_names(self):
		return ['Next_%s' % self.LTV]

	def get_id_names(self):
		return [self.feature_generator.cus_id, self.feature_generator.Customer_Age]

	def get_group_name(self):
		return [self.group_ltv]

	def get_constants(self):
		return {'activedate_mapping': self.feature_generator.activedate_mapping, \
				'ltvrate_byage': self.ltvrate_byage}
		
	def set_constants(self, constants):
		self.feature_generator.activedate_mapping = constants['activedate_mapping']
		self.ltvrate_byage = constants['ltvrate_byage']

	def _get_trackingdata_byage(self, df_preprocessed):
		tracking_data = self.feature_generator.create_tracking_bytarget(df_preprocessed, cal_LTV, self.LTV)
		if self.ltvrate_byage is None:
			tracking_data['ltv_rate'] = tracking_data['Next_%s' %self.LTV]/tracking_data[self.LTV]
			self.ltvrate_byage = tracking_data.groupby([self.feature_generator.Customer_Age])['ltv_rate'].mean().reset_index(name = 'LTV_rate')
			del tracking_data['ltv_rate'] 
			self.ltvrate_byage['LTV_rate'] = self.ltvrate_byage['LTV_rate'].fillna(self.ltvrate_byage['LTV_rate'].mean())

		tracking_data = tracking_data.merge(self.ltvrate_byage, on = self.feature_generator.Customer_Age)
		return tracking_data

	def create_training_data(self, tracking_data, profile_features):
		training_data = pd.merge(tracking_data, profile_features, on = self.feature_generator.keylist, how = 'left')
		training_data['key'] = training_data.apply(lambda r: \
					'%s_%d' % (r[self.feature_generator.cus_id], r[self.feature_generator.Customer_Age]), axis = 1)
		for col in [self.feature_generator.Customer_Age]:#, self.feature_generator.InvoiceDate_Range]:
			training_data[col] = training_data[col] + 1
		arr = np.array(np.log(training_data[self.LTV])).reshape(-1, 1)
		training_data[self.group_ltv] = compute_clusters(arr, num_clusters = self.num_clusters)
		training_data[self.group_ltv] = _generate_sortedcluster(training_data[self.LTV], training_data[self.group_ltv], "mean")		
		return training_data
		
	def prepare_data(self, df):
		df2 = self.feature_generator.preprocess(df)
		self.feature_generator.update_cusinfo(df2) # for both train/test
		df2 = self.feature_generator.add_PurchaseFeatures(df2)
		tracking_data = self._get_trackingdata_byage(df2)
		profile_features = self.feature_generator.create_profile(df2)
		training_data = self.create_training_data(tracking_data, profile_features)
		return training_data