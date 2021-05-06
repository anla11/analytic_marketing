import numpy as np
import pandas as pd

AGG_METRIC = ['max', 'mean', 'delta_prob', 'delta_cor_local']

class FeatureImpact():
	def __init__(self, percentiles = [0.2, 0.8]):
		self.percentiles = percentiles
		self.target_name, self.feature_name = None, None
		self.grtarget_name, self.grfeature_name = None, None
		self.grtarget_labels, self.grfeature_labels = None, None
		self.groupdf = None, None

	# def _get_groupdata(self, values, percentiles = [0.2, 0.8]): #values: series in pandas
	#     stats_list = np.array([np.percentile(values, q=q*100) for q in percentiles]) 
	#     tmp = values.apply(lambda r: np.sum([r>stats for stats in stats_list]))
	#     shift_group = dict(zip(sorted(tmp.unique()), range(tmp.nunique())))
	#     groups = tmp.apply(lambda r: shift_group[r])
	#     return groups

	def set_data(self, data, target_name, feature_name):
		from model.utils import _get_groupdata

		self.target_name, self.feature_name = target_name, feature_name
		self.grtarget_name, self.grfeature_name = '%s_Group' % target_name, '%s_Group' % feature_name
		
		self.groupdf = data[[target_name, feature_name]].copy()
		self.groupdf[self.grtarget_name] = _get_groupdata(self.groupdf[self.target_name], self.percentiles)
		self.groupdf[self.grfeature_name] = _get_groupdata(self.groupdf[self.feature_name], self.percentiles)
	
		self.grtarget_labels, self.grfeature_labels = self.groupdf[self.grtarget_name].unique(), self.groupdf[self.grfeature_name].unique()

	# Create Matrix Impact of each group of feature on each group of target
	def _cal_matrix_groupimpact_byprob(self):
		matrix = np.zeros([len(self.grtarget_labels), len(self.grfeature_labels)])
		for row in range(len(self.grtarget_labels)):
			for col in range(len(self.grfeature_labels)):
				matrix[row,col] = len(self.groupdf[(self.groupdf[self.grtarget_name]==self.grtarget_labels[row])&(self.groupdf[self.grfeature_name]==self.grfeature_labels[col])])/len(self.groupdf[self.groupdf[self.grfeature_name]==self.grfeature_labels[col]])
		return matrix
	
	# def _cal_correlation(self, x, y):
	# 	from scipy.stats import pearsonr as ps
	# 	cor, conf = ps(x, y)
	# 	return cor, conf

	def cal_groupimpact_byprob(self):
		matrix  = self._cal_matrix_groupimpact_byprob()
		cmp = np.max((matrix), axis = 1) - np.min((matrix), axis = 1)
		impact_byprob = {'target':self.target_name, 'feature': self.feature_name, 'vector_cmpprob':cmp, 'mean':np.mean(cmp), 'max':np.max(cmp), 'delta_prob':np.max(cmp)-np.min(cmp)}
		return impact_byprob 

	def _cal_matrix_groupimpact_bycor(self):
		from model.utils import cal_correlation 
		
		matrix_shape = [len(self.grtarget_labels), len(self.grfeature_labels)]
		cor_matrix, conf_matrix = np.zeros(matrix_shape), np.zeros(matrix_shape)

		cor_list, conf_list = [], [] 
		for label_idx in range(len(self.grtarget_labels)):
			df_grtarget = self.groupdf[self.groupdf[self.grtarget_name]==self.grtarget_labels[label_idx]]
			x, y = np.log(df_grtarget[self.target_name]), np.log(df_grtarget[self.feature_name])
			cor, conf = cal_correlation(x, y)        
			cor_list.append(cor)
			conf_list.append(conf)
			for feature_idx in range(len(self.grfeature_labels)):
				df_grfeature = df_grtarget[df_grtarget[self.grfeature_name]==self.grfeature_labels[feature_idx]]
				cor_matrix[label_idx, feature_idx], conf_matrix[label_idx, feature_idx] = np.nan, np.nan
				if len(df_grfeature) > 1:
					x_group, y_group = np.log(df_grfeature[self.target_name]), np.log(df_grfeature[self.feature_name])
					cor, conf = cal_correlation(x_group, y_group)
					cor_matrix[label_idx, feature_idx], conf_matrix[label_idx, feature_idx] = cor, conf
		return cor_list, conf_list, cor_matrix, conf_matrix

	def cal_groupimpact_bycor(self):
		cor_list, conf_list, cor_matrix, conf_matrix = self._cal_matrix_groupimpact_bycor()
		cmp_cor = np.nanmax((cor_matrix), axis = 1) - np.nanmin((cor_matrix), axis = 1)
		impact_bycor = {'target':self.target_name, 'feature': self.feature_name, 'vector_cor': np.array(cor_list), 'delta_cor_local': np.max(cmp_cor)-np.min(cmp_cor), 'vector_conf':np.array(conf_list), 'matrix_cor':cor_matrix, 'matrix_conf': conf_matrix}
		return impact_bycor

	def cal_featureimpact_bygroup(self):
		impact_byprob = self.cal_groupimpact_byprob()
		impact_bycor = self.cal_groupimpact_bycor()
		impact = {}
		impact.update(impact_byprob)
		impact.update(impact_bycor)
		return impact  

	def cal_featureimpact(self, data, target_name, list_features):
		impacts = []
		for feature in list_features:
			self.set_data(data, target_name, feature)
			impact = self.cal_featureimpact_bygroup()
			impacts.append(impact)
		return impacts

	def parsing_featureimpact(self, feature_impact_list):
		feature_impact_df = pd.DataFrame({'Target':[], 'Feature': [], 'Summary_Metric':[], 'Impact':[]})  
		for feature_impact in feature_impact_list:
			feature_name = [feature_impact['feature']]*len(AGG_METRIC)
			target_name = [feature_impact['target']]*len(AGG_METRIC)
			impact_values = [feature_impact[i] for i in AGG_METRIC]
			feature_impact_df = feature_impact_df.append(pd.DataFrame({'Target':target_name, 'Feature': feature_name, 'Summary_Metric':AGG_METRIC, 'Impact':impact_values}))  
			vector_cor = feature_impact['vector_cor'][1:]
			feature_impact_df = feature_impact_df.append(pd.DataFrame({'Target':[feature_impact['target']], 'Feature': [feature_impact['feature']], 'Summary_Metric':['delta_cor'], 'Impact':[np.max(vector_cor)-np.min(vector_cor)]}))  
		return feature_impact_df.reset_index().drop(['index'], axis=1)        