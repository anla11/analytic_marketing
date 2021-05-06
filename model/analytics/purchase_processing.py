import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import datetime as dt
import numpy as np
from model.utils import generate_features, extract_relativedate, extract_daterange, get_ages, get_first_dates

def cal_LTV(df, keys, groups, feature_names, target_name):
	ltv_df = df.groupby(keys + groups)[feature_names[0]].sum().reset_index(name = target_name)
	return ltv_df

def cal_OrderDateFeature(data, keylist = ['CustomerID'], Recency_name = 'Recency'):
	def _cal_slope(x, y):
		from scipy.stats import linregress
		slope, intercept, r_value, p_value, std_err = linregress(x, y)
		return slope
	def _cal_momentum(dates):
		return _cal_slope(-dates, range(len(dates)))
	def _cal_AvgBetweenOrders(dates):
		return _cal_slope(range(len(dates)), -dates)	

	customer_orderdate = data.sort_values(by = Recency_name, ascending = False).groupby(keylist)[Recency_name].unique().reset_index(name = 'List_Recency')
	customer_orderdate.loc[:, 'AvgTimeBetweenOrder'] = customer_orderdate['List_Recency'].apply(lambda r: _cal_AvgBetweenOrders(r))
	customer_orderdate.loc[:, 'OrderMomentum'] = customer_orderdate['List_Recency'].apply(lambda r: _cal_momentum(r))
	del customer_orderdate['List_Recency']
	for key in ['AvgTimeBetweenOrder', 'OrderMomentum']:
		m = customer_orderdate[key].mean()
		if np.isnan(m):
			m = 1.0
		customer_orderdate[key] = customer_orderdate[key].fillna(m)
	return customer_orderdate


class Abstract_PurchaseFeatures():
	def __init__(self, data_config):
		self.cus_id       = data_config.get_column_names(key='user_id')      # 'CustomerID' 
		self.product_id   = data_config.get_column_names(key='product_id')   # 'StockCode'
		self.invoice_no   = data_config.get_column_names(key='invoice_no')   # 'InvoiceNo' 
		self.invoice_date = data_config.get_column_names(key='invoice_date') # 'InvoiceDate'
		self.num_units    = data_config.get_column_names(key='num_units')    # 'Quantity'
		self.unit_price   = data_config.get_column_names(key='unit_price')   # 'UnitPrice'

		self.keylist     = [self.cus_id]
		self.featurelist = []

	def _get_generatefeature_config(self):
		pass

class PurchaseFeatures(Abstract_PurchaseFeatures):
	def __init__(self, data_config):
		super().__init__(data_config)
		self.TotalPurchase       = 'TotalPurchase'		
		self.AvgTimeBetweenOrder = 'AvgTimeBetweenOrder'
		self.OrderMomentum       = 'OrderMomentum'
		self.LastPurchase        = 'LastPurchase'
		self.MinOrder            = 'MinOrder'
		self.MaxOrder            = 'MaxOrder'
		self.NumInvoice          = 'NumInvoice'

		self._Recency              = 'Recency'
		self.AvgPurchasePerInvoice = 'AvgPurchasePerInvoice'
		self.AvgQuantityPerInvoice = 'AvgQuantityPerInvoice'
		self.AvgProductPerInvoice  = 'AvgProductPerInvoice'

		self.keylist = [self.cus_id]
		self.featurelist = [self.TotalPurchase, self.NumInvoice, self.LastPurchase, \
							self.MinOrder, self.MaxOrder, self.OrderMomentum, self.AvgTimeBetweenOrder, \
							self.AvgPurchasePerInvoice, self.AvgQuantityPerInvoice, self.AvgProductPerInvoice]
		
	def _cast_type(self, df):
		# Types
		df.loc[:, self.num_units] = df[self.num_units].astype(int)
		df.loc[:, self.unit_price] = df[self.unit_price].astype(float)
		df.loc[:, self.invoice_date] = pd.to_datetime(df[self.invoice_date])
		return df

	def _remove_invalid(self, df, mode = 'training'):
		if mode == 'training':
			df = df.drop_duplicates()
			df = df[df[self.num_units]>0]
			df = df[df[self.unit_price]>0]
		return df

	def preprocess(self, df):
		df2 = df.dropna(subset = [self.cus_id])
		if self.num_units is None:
			self.num_units = 'Quantity'
			df2.loc[:, 'Quantity'] = [1] * len(df2)
		if self.unit_price is None:
			self.unit_price = 'UnitPrice'
			df2.loc[:, 'UnitPrice'] = [1] * len(df2)

		df2 = self._cast_type(df2)
		df2 = self._remove_invalid(df2)
		return df2 	

	def add_PurchaseFeatures(self, df):
		df.loc[:, self.TotalPurchase] = df[self.num_units]*df[self.unit_price]
		date_max = df[self.invoice_date].max()
		df.loc[:, self._Recency] = df[self.invoice_date].apply(lambda date: (date_max - date).days)
		return df

	def _generate_behavior(self, df):
		StatFeature_Define, HighLevelFeature_Define = self._get_generatefeature_config()
		stat_features = generate_features(df, keylist = self.keylist, StatFeature_Define=StatFeature_Define, HighLevelFeature_Define=HighLevelFeature_Define)
		
		order_features = cal_OrderDateFeature(df, keylist = self.keylist, Recency_name = self._Recency)
		behaviour_features = pd.merge(stat_features, order_features, how = 'inner', on = self.keylist)
		return behaviour_features

	def create_profile(self, df):
		profile = self._generate_behavior(df)
		feature_names = list(set(profile.columns) - set(self.keylist))
		profile = profile[self.keylist + feature_names]
		return profile	 

	def _get_generatefeature_config(self):
		StatFeature_Define = [ \
			{'key': self.keylist, 'data':self.invoice_no, 'agg_metric':'nunique', 'value_name':self.NumInvoice},\
			{'key': self.keylist, 'data':self.TotalPurchase, 'agg_metric':'sum', 'value_name':self.TotalPurchase}, \
			{'key': self.keylist, 'data':self.TotalPurchase, 'agg_metric':'min', 'value_name':self.MinOrder}, \
			{'key': self.keylist, 'data':self.TotalPurchase, 'agg_metric':'max', 'value_name':self.MaxOrder}, \
			{'key': self.keylist, 'data':self._Recency, 'agg_metric':'min', 'value_name':self.LastPurchase}, \
			]

		HighLevelFeature_Define = [
			{'key1': self.keylist, 'key2':[self.invoice_no], 'data':self.TotalPurchase, 'agg_metric_1':'mean', 'agg_metric_2':'sum', 'value_name':self.AvgPurchasePerInvoice}, \
			{'key1': self.keylist, 'key2':[self.invoice_no], 'data':self.num_units, 'agg_metric_1':'mean', 'agg_metric_2':'sum', 'value_name':self.AvgQuantityPerInvoice}, \
			{'key1': self.keylist, 'key2':[self.invoice_no], 'data':self.product_id, 'agg_metric_1':'mean', 'agg_metric_2':'count', 'value_name':self.AvgProductPerInvoice}]
		return StatFeature_Define, HighLevelFeature_Define

class PurchaseFeatures_ByAge(PurchaseFeatures):
	def __init__(self, data_config, activedate_mapping = None, rangelen = 30):
		super().__init__(data_config)
		self.rangelen =  rangelen 

		self.activedate_mapping = activedate_mapping 	# Active dates of customers. Can assign this value externally. 
														# If None, auto assign it as first active dates of customers
		self._ActiveDays = 'ActiveDays'
		self.Customer_Age       = 'Customer_Age'
		self.Customer_NextAge   = 'Customer_NextAge'
		self.keylist = [self.cus_id, self.Customer_Age]

	def cal_target(self, func, df, keys, groups, target_name):
		return func(df, keys, groups, self.featurelist, target_name)

	# def _cal_InvAge(self, df): #extract age of customer for each invoice
	# 	date_df = df[[self.cus_id, self.invoice_date]]
	# 	ages = get_ages(date_df, self.cus_id, self.invoice_date, activedate_mapping = self.activedate_mapping)  
	# 	return extract_daterange(ages, self.rangelen, recent_order = False)

	def add_PurchaseFeatures(self, df):
		df = super().add_PurchaseFeatures(df)
		df.loc[:, self._ActiveDays] = get_ages(df, self.cus_id, self.invoice_date, activedate_mapping = self.activedate_mapping)       
		df.loc[:, self.Customer_Age] = extract_daterange(df[self._ActiveDays], self.rangelen, recent_order = False)
		return df
		
	def _get_invoices_byage(self, date_df, age):
		invidx_byage = date_df[date_df[self.Customer_Age] == age].drop_duplicates([self.cus_id, self.invoice_date])[[self.cus_id, self.Customer_Age, self.invoice_date]]
		return invidx_byage

	def _get_data_byage(self, data, date_df, age):	#get all data of customer when they were at an age	
		invidx_df = self._get_invoices_byage(date_df, age) #all invoices
		data_byage = data.merge(invidx_df, on = [self.cus_id, self.invoice_date])
		return data_byage	

	def _get_interaction_byage(self, data, date_df, age, target_func, target_name): # get target values from all invoices when they were at an age
		data_byage = self._get_data_byage(data, date_df, age)
		target_byage_df = self.cal_target(func = target_func, df = data_byage, keys = [self.cus_id], groups = [self.Customer_Age], target_name = target_name)
		return target_byage_df		
	
	def update_cusinfo(self, df):
		date_df = df[[self.cus_id, self.invoice_date]]
		if self.activedate_mapping is None:
			self.activedate_mapping = get_first_dates(date_df, self.cus_id, self.invoice_date)
		old_cus = self.activedate_mapping.keys()
		new_cus_df = date_df[~date_df[self.cus_id].isin(old_cus)]
		if len(new_cus_df) > 0:
			new_activedate_mapping = get_first_dates(new_cus_df, self.cus_id, self.invoice_date)
			self.activedate_mapping.update(new_activedate_mapping)

	def _generate_behavior_byage(self, df): #extract behaviour features of customers when they were at ages
		StatFeature_Define, HighLevelFeature_Define = self._get_generatefeature_config()
		stat_features = generate_features(df, keylist = self.keylist, StatFeature_Define=StatFeature_Define, HighLevelFeature_Define=HighLevelFeature_Define)
		last_purchase_df = df.groupby(self.keylist).agg('max')[self._ActiveDays].reset_index(name = self.LastPurchase)
		stat_features = stat_features.merge(last_purchase_df, how = 'left', on=self.keylist)
		stat_features.loc[:, self.LastPurchase] = stat_features[self.LastPurchase]- stat_features[self.Customer_Age]*self.rangelen

		order_features = cal_OrderDateFeature(df, keylist = self.keylist)
		behaviour_features = pd.merge(stat_features, order_features, how = 'inner', on = self.keylist)
		return behaviour_features

	def create_tracking_idx(self, df): # create list values of keys for creating profile
		date_df = df[[self.cus_id, self.invoice_date, self.Customer_Age]]
		all_idx = []
		max_age = date_df[self.Customer_Age].max()
		for age in range(max_age+1):
			idx_df = self._get_invoices_byage(date_df, age)[[self.cus_id, self.Customer_Age]].drop_duplicates()
			all_idx.append(idx_df)
		tracking_idx = pd.concat(all_idx)
		tracking_idx.rename(columns = {self.Customer_Age: self.Customer_Age}, inplace = True)
		return tracking_idx

	def create_profile(self, df_preprocessed):
		tmp = df_preprocessed.copy()
		profile = self._generate_behavior_byage(tmp)
		feature_names = list(set(profile.columns) - set(self.keylist))
		profile = profile[self.keylist + feature_names]
		return profile	   

	def create_tracking_bytarget(self, df, target_func, target_name):
		target_df = self.cal_target(func = target_func, df = df, keys = [self.cus_id, self.invoice_no, self.invoice_date], groups = [], target_name = self.TotalPurchase)
		date_df = df[[self.cus_id, self.invoice_date, self.Customer_Age]]

		all_1age_df = []
		max_age = date_df[self.Customer_Age].max()
		for start_age in range(max_age+1):
			start_df = self._get_interaction_byage(target_df, date_df, start_age, target_func, target_name)
			end_df = self._get_interaction_byage(target_df, date_df, start_age + 1, target_func, target_name).rename(columns = {self.Customer_Age: self.Customer_NextAge, target_name: 'Next_%s' % target_name})
			# create a training row: customerid, current_age, next_age, current ltv, next_ltv
			df_temp = pd.merge(start_df, end_df, how = 'left', on = [self.cus_id])
			all_1age_df.append(df_temp[[self.cus_id, self.Customer_Age, self.Customer_NextAge, target_name, 'Next_%s' % target_name]])
		tracking_data = pd.concat(all_1age_df)
		return tracking_data		
 
	def _get_generatefeature_config(self):
		StatFeature_Define = [ \
			{'key': self.keylist, 'data':self.invoice_no, 'agg_metric':'nunique', 'value_name':self.NumInvoice},\
			{'key': self.keylist, 'data':self.TotalPurchase, 'agg_metric':'sum', 'value_name':self.TotalPurchase}, \
			{'key': self.keylist, 'data':self.TotalPurchase, 'agg_metric':'min', 'value_name':self.MinOrder}, \
			{'key': self.keylist, 'data':self.TotalPurchase, 'agg_metric':'max', 'value_name':self.MaxOrder}, \
			# {'key': self.keylist, 'data':self._Recency, 'agg_metric':'min', 'value_name':self.LastPurchase}, \
			]

		HighLevelFeature_Define = [
			{'key1': self.keylist, 'key2':[self.invoice_no], 'data':self.TotalPurchase, 'agg_metric_1':'mean', 'agg_metric_2':'sum', 'value_name':self.AvgPurchasePerInvoice}, \
			{'key1': self.keylist, 'key2':[self.invoice_no], 'data':self.num_units, 'agg_metric_1':'mean', 'agg_metric_2':'sum', 'value_name':self.AvgQuantityPerInvoice}, \
			{'key1': self.keylist, 'key2':[self.invoice_no], 'data':self.product_id, 'agg_metric_1':'mean', 'agg_metric_2':'count', 'value_name':self.AvgProductPerInvoice}]
		return StatFeature_Define, HighLevelFeature_Define		