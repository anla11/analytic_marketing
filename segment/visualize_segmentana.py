# 1. Visualize Boxplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from model.utils import get_summarizegroup, get_clustersorted

def vis_groups_byclusters(data, target_name, feature_name, sort_metric='mean'):
	#   cluster_sorted = data.groupby(['%s_Cluster'%target_name])[target_name].mean().reset_index().sort_values(by=target_name)['%s_Cluster'%target_name]
	cluster_sorted = get_clustersorted(data, cluster_name = '%s_Cluster'%target_name, valuesorted_name=target_name, sort_metric=sort_metric)
	vis_df = data[(data['%s_Cluster'%target_name].isin(cluster_sorted))]
	sns.boxplot(x=vis_df['%s_Cluster'%target_name], y=vis_df[target_name], fliersize=2, order=cluster_sorted)
	plt.show()
	order = data.groupby(['%s_Cluster'%target_name]).agg(sort_metric)[feature_name].reset_index().sort_values(by=feature_name)['%s_Cluster'%target_name]
	sns.boxplot(x=data['%s_Cluster'%target_name], y=data[feature_name], fliersize=2, order=order)
	plt.show()

# 2. Visualize Heatmap
def plot_histogram_matrixcluster(matrix_clusters, generic_group_define, func = np.sum):
	cnt_genericgroup = np.zeros((len(generic_group_define)-1))
	for i in range(len(generic_group_define)-1):
		start = generic_group_define[i]
		end = generic_group_define[i+1]
		cnt_genericgroup[i] = func(matrix_clusters[start[0]:end[0], start[1]:end[1]])
	plt.bar(x=range(len(cnt_genericgroup)), height = cnt_genericgroup, width=0.5)
	plt.xticks(range(len(cnt_genericgroup)))

def visualize_matrixcluster(cluster_df, target_name, generic_group_define = [], func_group_name = "mean", func_sum = np.sum, histogram_vis=True, sort_metric='mean', heatmap_annot = False, kpi_name = ("TotalPrice", "AvgPricePerInvoice")):
	kpi1_name, kpi2_name = kpi_name
	kpi1_name_group = "%s_Cluster"%kpi1_name
	kpi2_name_group = "%s_Cluster"%kpi2_name
	kpi1cluster_sorted = get_clustersorted(cluster_df, cluster_name = kpi1_name_group, valuesorted_name=kpi1_name, sort_metric=sort_metric)
	kpi2cluster_sorted = get_clustersorted(cluster_df, cluster_name = kpi2_name_group, valuesorted_name=kpi2_name, sort_metric=sort_metric)
	mean_hybridgroup = cluster_df[[kpi1_name_group, kpi2_name_group, target_name]].groupby([kpi1_name_group, kpi2_name_group]).agg({target_name:func_group_name}).reset_index()
	mean_hybridgroup['KPI1_Cluster_pos'] = mean_hybridgroup[kpi1_name_group].apply(lambda r: np.where(np.array(r)==kpi1cluster_sorted)[0][0])
	mean_hybridgroup['KPI2_Cluster_pos'] = mean_hybridgroup[kpi2_name_group].apply(lambda r: np.where(np.array(r)==kpi2cluster_sorted)[0][0])

	matrix_clusters = np.zeros((cluster_df[kpi1_name_group].nunique(), cluster_df[kpi2_name_group].nunique()))
	idy = mean_hybridgroup.KPI1_Cluster_pos.nunique()-1-np.array(range(mean_hybridgroup.KPI1_Cluster_pos.nunique()))
	matrix_clusters[mean_hybridgroup.KPI1_Cluster_pos.nunique()-1-mean_hybridgroup.KPI1_Cluster_pos, mean_hybridgroup.KPI2_Cluster_pos] = np.array(mean_hybridgroup[target_name]).astype(int)
	sns.heatmap(matrix_clusters,  annot=heatmap_annot, cmap="Blues", yticklabels=idy)
	plt.title('Distribution of %s (color) on groups of %s and %s' %(target_name, kpi1_name, kpi2_name))
	plt.xlabel(kpi2_name)
	plt.ylabel(kpi1_name)
	plt.show()

	if histogram_vis:
		matrix_clusters = np.zeros((cluster_df[kpi1_name_group].nunique(), cluster_df[kpi2_name_group].nunique()))
		matrix_clusters[mean_hybridgroup.KPI1_Cluster_pos, mean_hybridgroup.KPI2_Cluster_pos] = np.array(mean_hybridgroup[target_name]).astype(int)
		plot_histogram_matrixcluster(matrix_clusters, generic_group_define, func_sum)
		plt.title('Histogram of %s by groups of %s and %s' %(target_name, kpi1_name, kpi2_name))
		plt.xlabel('Groups of pair (%s and %s)'%(kpi1_name_group, kpi2_name_group))
		plt.show()

def visualize_group_effect(GroupFeature_df, target_names, feature_cols, group_name, sort_metric = 'median', figsize = (25,10)):
	group_sum_df = get_summarizegroup(GroupFeature_df, group_name, target_names, sort_metric)
	fig, axs = plt.subplots(len(target_names), len(feature_cols) + 1, figsize = figsize)
	for t in range(len(target_names)):
		target_name = target_names[t]
		group_sorted = group_sum_df.sort_values(by=target_name)[group_name]
		sns.boxplot(x = GroupFeature_df[group_name], y = np.log(GroupFeature_df[target_name]), fliersize=0.5, order = group_sorted, ax=axs[t][0])
		for f in range(len(feature_cols)):
			feature = feature_cols[f]
			sns.boxplot(x = GroupFeature_df[group_name], y = np.log(GroupFeature_df[feature]), fliersize=0.5, order = group_sorted, ax = axs[t][f+1])
	plt.show()


def vis_ClusterReal(tp_values, target):
	sns.scatterplot(x =tp_values[target], y = tp_values['%s_ClusterUnit'%target], hue = tp_values['%s_Cluster'%target])
	plt.legend(bbox_to_anchor=(1, 0.5))
	plt.show()
	sns.scatterplot(x =tp_values[target],  y = tp_values['%s_ClusterReal'%target], hue = tp_values['%s_Cluster'%target])
	plt.legend(bbox_to_anchor=(1, 0.5))
	plt.show()
	sns.boxplot(x =tp_values['%s_Cluster'%target], y = tp_values['%s_ClusterReal'%target], fliersize=1.0)
	plt.show()
	sns.boxplot(x =tp_values['%s_Cluster'%target], y = tp_values[target], fliersize=0.1)
	plt.show()    