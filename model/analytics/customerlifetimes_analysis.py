import pandas as pd 
import numpy as np 

import lifetimes

class CustomerLifetimes_Analysis():
	def __init__(self, data_config):
		self.cus_id = 'CustomerID'#data_config.get_column_names(key='user_id')
		self.invoice_date = 'InvoiceDate'#data_config.get_column_names(key='invoice_date')
		self.invoice_no = 'InvoiceNo'#data_config.get_column_names(key='invoice_no')
		self.num_units = 'Quantity'#data_config.get_column_names(key='num_units')
		self.unit_price = 'UnitPrice'#data_config.get_column_names(key='unit_price')
		self.amount = 'SalesAmount'
		self.bgf, self.ggf = None, None 
		self.filtered_df = None 


	def __compute_customer_metrics(self, data_df):
		data_df[self.amount] = data_df[self.num_units] * data_df[self.unit_price]
		current_date = data_df[self.invoice_date].max()

		metrics_df = (
			lifetimes.utils.summary_data_from_transaction_data(data_df,
				customer_id_col=self.cus_id,
				datetime_col=self.invoice_date,
				observation_period_end = current_date, 
				freq='D',
				monetary_value_col= self.amount  
			))

		filtered_df = metrics_df[metrics_df['frequency'] > 0]
		return filtered_df

	def fit_bgf(self, data_df):
		data_df = self.__compute_customer_metrics(data_df)
		self.bgf = lifetimes.fitters.beta_geo_fitter.BetaGeoFitter(penalizer_coef=0.0)
		self.bgf.fit(frequency=data_df['frequency'], recency=data_df['recency'], T=data_df['T'], weights=None)

	def fit_ggf(self, data_df):
		data_df = self.__compute_customer_metrics(data_df)
		self.ggf = lifetimes.fitters.gamma_gamma_fitter.GammaGammaFitter()
		self.ggf.fit(data_df['frequency'], data_df['monetary_value'])

	def parse_customer_metrics(self, data_df):
		metrics_df = self.__compute_customer_metrics(data_df)
		return metrics_df[['frequency']], metrics_df[['recency']], metrics_df[['T']], metrics_df[['monetary_value']]

	# def get_monetary_value(self, data_df):
	# 	data_df = self.__compute_customer_metrics(data_df)
	# 	return data_df[['monetary_value']]

	def get_prob_alive(self, data_df):
		if (self.bgf is None):
			print ('You need to call fit_bgf first.')
			return None

		data_df = self.__compute_customer_metrics(data_df)
		data_df['prob_alive']=self.bgf.conditional_probability_alive(data_df['frequency'], data_df['recency'], data_df['T'])
		return data_df[['prob_alive']]

	def get_purchases_next_n_days(self, data_df, n_day):
		if (self.bgf is None):
			print ('You need to call fit_bgf first.')
			return None

		data_df = self.__compute_customer_metrics(data_df)
		data_df['purchases_next_%s_days'%str(n_day)] = (
			self.bgf.conditional_expected_number_of_purchases_up_to_time(n_day, data_df['frequency'], data_df['recency'], data_df['T']))
		return data_df[['purchases_next_%s_days'%str(n_day)]]

	def get_clv(self, data_df):
		if (self.bgf is None):
			print ('You need to call fit_bgf first.')
			return None

		if (self.ggf is None):
			print ('You need to call fit_ggf first.')
			return None

		clv_input_pd = self.__compute_customer_metrics(data_df)
		clv_input_pd['clv'] = (
			self.ggf.customer_lifetime_value(self.bgf, clv_input_pd['frequency'], clv_input_pd['recency'], clv_input_pd['T'], clv_input_pd['monetary_value'], time=12, discount_rate=0.01 
		))

		return clv_input_pd[['clv']]
