import numpy as np
import pandas as pd

from propensity_prediction.tasks.abstract import Abstract_Context

class NextAction_Context(Abstract_Context):
	def get_label_names(self):
		raise NotImplementedError()  
	def get_convfeature_names(self):
		raise NotImplementedError()
	def get_feature_names(self):
		raise NotImplementedError()
	def prepare_data(self):
		raise NotImplementedError()

class NextAction_InSession_Context(NextAction_Context):
	def __init__(self, data_config):
		self.data_config = data_config
		self.user_id = data_config.get_column_names(key='user_id')
		self.product_id = data_config.get_column_names(key='product_id')
		self.user_session = data_config.get_column_names(key='user_session')
		self.event = data_config.get_column_names(key='event')
		self.order_actions = data_config.get_column_names(key='event_constraint') 
		self.other_features = data_config.get_column_names(key='metadata')
		self.convfeature_names, self.feature_names, self.label_names = None, None, None

	def copy(self):
		obj = NextAction_InSession_Context(self.data_config)  
		return obj  
	
	def __generate_label_names(self):
		labels = []
		for order_action in self.order_actions:
			label = '%s2%s' % (order_action['source'], order_action['des'])
			labels.append(label)
		return labels          

	def __generate_convfeature_names(self):
		features = []
		for order_action in self.order_actions:
			feature_user = 'user_rate_' + order_action['source'][0] + '2' + order_action['des'][0]
			feature_product = 'product_rate_' + order_action['source'][0] + '2' + order_action['des'][0]
			features.append(feature_user)
			features.append(feature_product)
		return features
    
	def __generate_feature_names(self):
		list_features = self.get_convfeature_names() + self.other_features
		return list_features
    
	def __converting_action_features_df(self, data_df, order_action):
		src = order_action['source']
		des = order_action['des']

		data_temp = data_df[data_df[self.event].isin([src, des])].loc[:,[self.user_id, self.product_id, self.user_session, self.event]]

		# Split event_type to 2 new attributes.
		data_temp[src] = np.where(data_temp[self.event] == src, 1, 0)
		data_temp[des] = np.where(data_temp[self.event] == des, 1, 0)
		new_data = data_temp.groupby([self.user_id, self.product_id]).sum().reset_index()	
		
		# Create label for new data 
		new_data['%s2%s' % (src, des )] = np.where(((new_data[src] > 0)&(new_data[des] > 0 )), 1, 0)
		
		# Create converting action features and merge them
		user_rate = new_data.groupby([self.user_id])['%s2%s' % (src, des)].mean().reset_index(name='user_rate_%s2%s' % (src[0], des[0]))
		product_rate = new_data.groupby([self.product_id])['%s2%s' % (src, des)].mean().reset_index(name='product_rate_%s2%s' % (src[0], des[0]))
		new_data = new_data.merge(user_rate, how='left')
		new_data = new_data.merge(product_rate, how='left')
		return new_data

	def get_convfeature_names(self):
		if self.convfeature_names is None:
			self.convfeature_names = self.__generate_convfeature_names()
		return self.convfeature_names

	def get_feature_names(self):
		if self.feature_names is None:
			self.feature_names = self.__generate_feature_names()
		return self.feature_names

	def get_label_names(self):
		if self.label_names is None:
			self.label_names = self.__generate_label_names()
		return self.label_names

	def get_id_names(self):
		return [self.user_id, self.product_id]

	def prepare_data(self, data_df):
		data = []
		for order_action in self.order_actions:
			data_temp = self.__converting_action_features_df(data_df, order_action)
			data.append(data_temp)
		# Merge new data are created above
		new_data = data[0]
		for i in range(len(self.order_actions)-1):
			new_data = new_data.merge(data[i+1], how='outer')
            
		if len(self.other_features) != 0:
			input_features_df = data_df[self.other_features + [self.user_id, self.product_id] ]
			new_data = new_data.merge(input_features_df, how='outer')
		return new_data     