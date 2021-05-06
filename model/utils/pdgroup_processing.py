import pandas as pd 
import numpy as np

def _get_groupdata(values, percentiles = [0.2, 0.8]): #values: series in pandas
	stats_list = np.array([np.percentile(values, q=q*100) for q in percentiles]) 
	tmp = values.apply(lambda r: np.sum([r>stats for stats in stats_list]))
	shift_group = dict(zip(sorted(tmp.unique()), range(tmp.nunique())))
	groups = tmp.apply(lambda r: shift_group[r])
	return groups

def get_groupdata(data, list_groupcol, percentiles = [0.2, 0.8]):
	group_df = pd.DataFrame()
	for col in list_groupcol:
		group_df['%s_Group' % col] = _get_groupdata(data[col], percentiles)
	return group_df 

def get_summarizegroup(df, group_name, target_names, sort_metric='mean'):
	group_sum_df = df.groupby([group_name]).agg(sort_metric)[target_names].reset_index()
	return group_sum_df    

def get_groupnamesorted(df, group_name = 'TotalPrice_Group', valuesorted_name = 'TotalPrice', sort_metric = 'mean'):
	name_sorted = get_summarizegroup(df, group_name, valuesorted_name, sort_metric=sort_metric).sort_values(by=valuesorted_name)
	return name_sorted[group_name]  

def cal_distribution_bygroup(df, group_name, value_name):
	mean_bygroup = get_summarizegroup(df, group_name = group_name, target_names=[value_name], sort_metric='mean').sort_values(by = value_name)
	mapping_mean_group = dict(zip(mean_bygroup[group_name], mean_bygroup[value_name]))
	std_bygroup = get_summarizegroup(df, group_name = group_name, target_names=[value_name], sort_metric='std').sort_values(by = value_name)
	mapping_std_group = dict(zip(std_bygroup[group_name], std_bygroup[value_name]))
	return {'mean':mapping_mean_group, 'std':mapping_std_group} 

def get_group_frompercentile(data, target_name, feature_name, percentiles = [0.2, 0.8]):
	grtarget_name = '%s_Group' % target_name
	grfeature_name = '%s_Group' % feature_name
	group_names = (grtarget_name, grfeature_name)

	groupdf = data[[target_name, feature_name]].copy()
	groupdf[grtarget_name] = _get_groupdata(groupdf[target_name], percentiles)
	groupdf[grfeature_name] = _get_groupdata(groupdf[feature_name], percentiles)

	grtarget_labels, grfeature_labels = groupdf[grtarget_name].unique(), groupdf[grfeature_name].unique()
	group_labels =  (grtarget_labels, grfeature_labels)

	return groupdf, group_names, group_labels