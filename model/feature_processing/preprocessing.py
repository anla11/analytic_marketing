from model.utils.transform import scale_minmax
import pandas as pd
import numpy as np

def format_df(df, feature_types):
	df_res = df.copy()
	cat_features = []
	for c, t in feature_types.items():
		if t == 'numeric':
			df_res[c] = np.array(pd.to_numeric(df_res[c], errors='coerce')).astype(float)
		if t == 'ordering':
			df_res[c] = np.array(df_res[c]).astype(float)
		if t=='numeric' or t=='ordering':
			df_res[c] = scale_minmax(df_res[c])
		if t=='category':
			del df_res[c]
			cat_features.append(c)
	for c in cat_features:
		cate_df = pd.get_dummies(df[c], dummy_na=False, drop_first=True)
		cate_df=cate_df.astype('int')
		cate_df.columns = ['%s_%s' % (c, v) for v in cate_df.columns]
		df_res = pd.concat([df_res, cate_df], axis=1)
	return df_res

def remove_objects(df, feature_types):
	rm_obj = []
	for c, t in feature_types.items():
		if t=='id' or t=='object' or t=='text':
			rm_obj.append(c)
			del df[c]
	for c in rm_obj:
		del feature_types[c]
	return df, feature_types

def numerize_category(df_col):
	df_col = np.array(df_col.astype('category').cat.codes).astype(float)
	return df_col

def auto_preprocess(df, key_types, feature_types, dropna = False):
	df2, feature_types_processed = remove_objects(df[feature_types], feature_types)
	df2 = format_df(df2, feature_types_processed)
	features = df2.columns
	for col, col_type in key_types.items():
		if col_type is None:
			df2[col] = df[col]
		if col_type=='numerize_category':
			df2[col] = numerize_category(df[col])
	if dropna:
		df2.dropna(inplace=True)
	new_feature_types = dict(zip(features, ['numeric']*len(features)))
	return df2, new_feature_types
