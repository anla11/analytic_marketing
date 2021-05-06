from propensity_prediction.auto_preprocess.abstract import Abstract_Preprocess
from model.utils import scale_minmax

import pandas as pd
import numpy as np


class Base_Preprocess(Abstract_Preprocess):
	def __init__(self, key_types, feature_types):
		self.data = None
		self.key_types = key_types
		self.feature_types = feature_types.copy()

	def _format_df(self):
		df_res = self.data.copy()
		cat_features = []
		for c, t in self.feature_types.items():
			if t == 'numeric':
				df_res[c] = np.array(pd.to_numeric(df_res[c], errors='coerce')).astype(float)
			if t == 'ordering':
				df_res[c] = np.array(df_res[c]).astype(float)
			if t == 'numeric' or t == 'ordering':
				df_res[c] = scale_minmax(df_res[c])
			if t == 'category':
				del df_res[c]
				cat_features.append(c)

		for c in cat_features:
			cate_df = pd.get_dummies(self.data[c], dummy_na=False, drop_first=True)
			cate_df = cate_df.astype('int')
			cate_df.columns = ['%s_%s' % (c, v) for v in cate_df.columns]
			df_res = pd.concat([df_res, cate_df], axis=1)
		return df_res

	def _remove_object(self):
		rm_obj = []
		for c, t in self.feature_types.items():
			if t == 'id' or t == 'object' or t == 'text':
				rm_obj.append(c)
				del self.data[c]
		for c in rm_obj:
			del self.feature_types[c]

	# def _numerize_category(self, df_col):
	# 	df_col = np.array(df_col.astype('category').cat.codes).astype(float)
	# 	return df_col

	def auto_preprocess(self, data_raw, dropna=False):
		self.data = data_raw[self.feature_types].copy()
		self._remove_object()
		self.data = self._format_df()
		features = self.data.columns
		for col, col_type in self.key_types.items():
			if col_type is None:
				self.data[col] = data_raw[col]
			if col_type == 'key':
				self.data[col] = data_raw[col] #self._numerize_category(data_raw[col])
		if dropna:
			self.data.dropna(inplace=True)
		self.feature_types = dict(zip(features, ['numeric'] * len(features)))
		return self.data, self.feature_types