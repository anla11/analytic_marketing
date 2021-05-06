import pandas as pd
import numpy as np
import datetime as dt 
from model.utils import generate_features


class InSessionFeatures():
	def __init__(self, data_config):
		self.product_id = data_config.get_column_names(key='product_id')
		self.user_id = data_config.get_column_names(key='user_id')
		self.user_session = data_config.get_column_names(key='user_session')
		self.event_time = data_config.get_column_names(key='event_time')
		self.event_type = data_config.get_column_names(key='event_type')
		self.price = data_config.get_column_names(key='price')

		self.duration = 'Duration'
		self.keylist = None
		self.stats_define = None

	def preprocess(self, df):
		data_df = df.copy()
		data_df[self.price] = data_df[self.price].astype(float)
		data_df[self.event_time] = data_df[self.event_time].apply(lambda r: pd._libs.tslibs.timestamps.Timestamp(r[:19]))
		return data_df

	def _get_features(self, data_df, keylist, stats_define):    
		for i in range(len(stats_define)):
			stats_define[i]['key'] = keylist    
		stats_df = generate_features(data_df.dropna(subset = keylist), keylist = keylist, StatFeature_Define = stats_define)
		return stats_df 

	def get_duration_df(self, data_df):
		start_df = data_df.groupby(self.keylist)[self.event_time].min().reset_index(name = 'StartTime')
		end_df = data_df.groupby(self.keylist)[self.event_time].max().reset_index(name = 'EndTime')
		dur_df = pd.merge(start_df, end_df, on = self.keylist)
		dur_df[self.duration] = (dur_df['EndTime'] - dur_df['StartTime']).dt.total_seconds()
		del dur_df['StartTime']
		del dur_df['EndTime']
		return dur_df

	def get_stats_features(self, data_df, get_duration = True):
		stats_df = self._get_features(data_df, self.keylist, self.stats_define)
		if get_duration:
			dur_df = self.get_duration_df(data_df)
			stats_df = stats_df.merge(dur_df, on = self.keylist)
		return stats_df

	def get_stats_features_byatt(self, data_df):
		stats_df = self._get_features(data_df, self.keylist + [self.event_type], self.stats_define)
		return stats_df


class InSessionFeatures_ByCus(InSessionFeatures):
	def __init__(self, data_config):
		super().__init__(data_config)
		self.product_count = 'ProductCount_UP'
		self.event_count = 'EventCount_UP'
		self.total_price = 'TotalPrice_UP'
		self.duration = 'Duration_UP'
		self.feature_cols = [self.product_count, self.event_count, self.total_price, self.duration]
		self.keylist = [self.user_id, self.user_session]
		self.stats_define = [ \
					{'key': self.keylist, 'data': self.product_id, 'agg_metric':'nunique', 'value_name': self.product_count},\
					{'key': self.keylist, 'data': self.product_id, 'agg_metric':'count', 'value_name': self.event_count}, \
					{'key': self.keylist, 'data': self.price, 'agg_metric':'sum', 'value_name': self.total_price}, \
					]

	def get_profile(self, data_df):
		stats_df = self.get_stats_features(data_df)
		profile = stats_df.groupby([self.user_id])[[self.product_count, self.event_count, self.total_price, self.duration]].mean().reset_index()
		return profile

	def get_profile_byevent(self, data_df):
		stats_byevent = self.get_stats_features_byatt(data_df)
		profile_byevent = stats_byevent.groupby([self.user_id, self.event_type])[[self.product_count, self.event_count, self.total_price]].mean().reset_index()
		return profile_byevent


class InSessionFeatures_ByProd(InSessionFeatures):
	def __init__(self, data_config):
		super().__init__(data_config)
		self.user_count = 'UserCount_PP'
		self.event_count = 'EventCount_PP'
		self.total_price = 'TotalPrice_PP'
		self.feature_cols = [self.user_count, self.event_count, self.total_price]
		self.keylist = [self.product_id, self.user_session]
		self.stats_define = [ \
					{'key': self.keylist, 'data': self.user_id, 'agg_metric':'nunique', 'value_name': self.user_count},\
					{'key': self.keylist, 'data': self.user_id, 'agg_metric':'count', 'value_name': self.event_count}, \
					{'key': self.keylist, 'data': self.price, 'agg_metric':'sum', 'value_name': self.total_price}, \
					]

	def get_profile(self, data_df, get_duration = False):
		stats_df = self.get_stats_features(data_df)
		profile = stats_df.groupby([self.product_id])[[self.user_count, self.event_count, self.total_price]].mean().reset_index()
		return profile

	def get_profile_byevent(self, data_df):
		stats_byevent = self.get_stats_features_byatt(data_df)
		profile_byevent = stats_byevent.groupby([self.product_id, self.event_type])[[self.user_count, self.event_count, self.total_price]].mean().reset_index()
		return profile_byevent


class InSessionFeatures_ByTransaction(InSessionFeatures):
	def __init__(self, data_config):
		super().__init__(data_config)
		self.event_count = 'EventCount_SS'
		self.total_price = 'TotalPrice_SS'
		self.duration = 'Duration_SS'
		self.feature_cols = [self.event_count, self.total_price, self.duration]
		self.keylist = [self.user_id, self.product_id, self.user_session]
		self.stats_define = [ \
					{'key': self.keylist, 'data': self.event_time, 'agg_metric':'count', 'value_name': self.event_count}, \
					{'key': self.keylist, 'data': self.price, 'agg_metric':'sum', 'value_name': self.total_price}, \
					]    

	def get_profile(self, data_df):
		stats_df = self.get_stats_features(data_df)
		return stats_df

	def get_profile_byevent(self, data_df):
		stats_byevent = self.get_stats_features_byatt(data_df)
		profile_byevent = pd.pivot_table(stats_byevent, index = self.keylist, columns = [self.event_type], aggfunc = np.sum).reset_index()
		return profile_byevent