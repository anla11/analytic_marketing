import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import datetime as dt
import numpy as np
from propensity_prediction.tasks.abstract import Abstract_Context

class LTVPrediction_Context(Abstract_Context):
	def __init__(self, data_config):
		self.cus_id = data_config.get_column_names(key='user_id')
		self.invoice_date = data_config.get_column_names(key='invoice_date')
		self.invoice_no = data_config.get_column_names(key='invoice_no')
		self.num_units = data_config.get_column_names(key='num_units')
		self.unit_price = data_config.get_column_names(key='unit_price')
		self.months_train = data_config.get_column_names(key='event_constraint')
		self.label_name = 'CLV' 
		self.list_months = None

	def get_feature_names(self):
		if self.months_train <= 1:
			num_months_train = int(self.months_train*len(self.list_months))
			return self.list_months[-num_months_train:]
		return self.list_months[-self.months_train:]

	def get_label_names(self):
		return [self.label_name]

	def get_id_names(self):
		return [self.cus_id]

	def prepare_data(self, data_df):
		df = data_df.copy()

		df['spent_money'] = df[self.unit_price]*df[self.num_units]
		df['month_yr'] = df[self.invoice_date].apply(lambda x: x.strftime('%Y-%m'))

		sale_df=df.pivot_table(index=[self.cus_id],columns=['month_yr'],values='spent_money',aggfunc='sum',fill_value=0).reset_index()
		sale_df[self.label_name]=sale_df.iloc[:,2:].sum(axis=1)	

		self.list_months = list(sale_df.columns[1:-1].values)

		return sale_df

