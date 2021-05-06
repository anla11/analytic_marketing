import pandas as pd

def _get_StatFeature(transaction_df, keylist, data_name, agg_metric='mean', value_name='value'):
	'''
	Getting summary of the information from transaction of items
	For example:
		Counting the number of invoices of users in a month
		Counting the number of purchases of items in a month
	keylist: ['CustomerID'] or ['InvoiceNo'] or ['CustomerID', 'InvoiceNo']
	data_name and sum_metric:
		data_name = 'InvoiceNo', sum_metric='count': Counting the number of invoice of all users/products
		data_name = 'Quantity', sum_metric='mean': average number of quantity that products are purchased
	'''
	return transaction_df.groupby(keylist).agg(agg_metric)[data_name].reset_index(name=value_name)

def _get_HighLevelFeature(transaction_df, key1, key2, data_name, agg_metric_key1='mean', agg_metric_key2='mean', value_name = 'value'):
	'''
	Getting summary of the information of key1 from list of values (data_name) of key2 in transaction
	For example: 
		key1='user_id' key2='item_id' data_name='invoicecount'
		u00                 iA          13
		u00                 iB          20
		u01                 iA          18
	Output
		key1='user_id'    colum_name=avg_byitem_invoicecount
		u00                 16.5
		u01                 18.0
	'''
	stat_key2 = _get_StatFeature(transaction_df, key2, data_name, agg_metric_key2, value_name=data_name)
	df_tmp = transaction_df[key1 + key2].drop_duplicates()    
	return df_tmp.merge(stat_key2, on = key2).groupby(key1).agg(agg_metric_key1)[data_name].reset_index(name=value_name)

def generate_features(data, keylist = ['CustomerID'], StatFeature_Define = {}, HighLevelFeature_Define = {}):
	df = data[keylist].drop_duplicates()
	for f in StatFeature_Define:
		feature_df = _get_StatFeature(data, keylist=f['key'], data_name=f['data'], agg_metric=f['agg_metric'], value_name=f['value_name'])
		df = df.merge(feature_df, on = keylist)
	for f in HighLevelFeature_Define:
		feature_df = _get_HighLevelFeature(data, f['key1'], f['key2'], f['data'], f['agg_metric_1'], f['agg_metric_2'], f['value_name'])
		df = df.merge(feature_df, on = keylist)        
	return df