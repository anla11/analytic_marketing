'''
	@author: anla-ds
	created date: 8 July, 2020
'''
import pandas as pd
import numpy as np
import lifelines

'''
	preprocess data to feed right-censored format of Survival Analysis here
'''

class Survival_Analysis():
	def __init__(self, data_config, sur_prob_col='survival_probabilities'):
		self.dur_col = data_config.get_column_names(key='user_seniority')
		self.event_col = data_config.get_column_names(key='event')
		self.feature_col = data_config.get_column_names(key='metadata')
		self.sur_prob_col = sur_prob_col
		self.kmf, self.cph = None, None
		self.kmf_survival_df, self.kmf_timeline_surprob_mapping = None, None
		self.timeline_keys = None

	def _fit_kmf(self, data_df):
		self.kmf = lifelines.KaplanMeierFitter()
		self.kmf.fit(durations=data_df[self.dur_col], event_observed=data_df[self.event_col], label=self.sur_prob_col)
		return self.kmf

	def __fill_timeline_surprob(self):
		self.timeline_keys = list(self.kmf_timeline_surprob_mapping.keys())
		all_keys = np.array(self.timeline_keys)
		np.sort(all_keys)

		left_idx, right_idx = 0, min(1, len(all_keys)-1)
		for k in range(all_keys[0], all_keys[-1]):
			if k == all_keys[right_idx]:
				left_idx = right_idx
				right_idx = min(left_idx + 1, len(all_keys)-1)
			if k == all_keys[left_idx]:
				continue
			if k - all_keys[left_idx] < (all_keys[right_idx] - all_keys[left_idx])/2:
				self.kmf_timeline_surprob_mapping[k] = self.kmf_timeline_surprob_mapping[all_keys[left_idx]]
			else:
				self.kmf_timeline_surprob_mapping[k] = self.kmf_timeline_surprob_mapping[all_keys[right_idx]]

	def __find_timeline_surprob(self, t):
		if t in self.timeline_keys:
			return self.kmf_timeline_surprob_mapping[t]
		else:
			all_keys = np.array(self.timeline_keys)
			np.sort(all_keys)			
			left_idx, right_idx = all_keys[0], all_keys[-1]
			if t <= left_idx: 
				return self.kmf_timeline_surprob_mapping[left_idx]
			return self.kmf_timeline_surprob_mapping[right_idx]
	
	def __parse_survival_kmf(self):
		if self.kmf is None:
			print ('You need to call _fit_kmf first.')
			return None
		if self.kmf_survival_df is None:
			self.kmf_survival_df = self.kmf.survival_function_.reset_index()
			self.kmf_survival_df[self.dur_col] = np.array(self.kmf_survival_df['timeline']).astype(int)
			del self.kmf_survival_df['timeline']
		if self.kmf_timeline_surprob_mapping is None:
			self.kmf_timeline_surprob_mapping = {}
		timeline_surprob_mapping = dict(zip(self.kmf_survival_df[self.dur_col], self.kmf_survival_df[self.sur_prob_col]))			
		self.kmf_timeline_surprob_mapping.update(timeline_surprob_mapping)
		self.__fill_timeline_surprob()
		return self.kmf_survival_df, self.kmf_timeline_surprob_mapping

	def get_cumulative_kmf_surprob(self, timepoint_arr):
		_, timeline_surprob_mapping = self.__parse_survival_kmf()
		return np.array(list(map(lambda x: self.__find_timeline_surprob(x), timepoint_arr)))

	def get_current_kmf_surprob(self, timepoint_arr):
		_, timeline_surprob_mapping = self.__parse_survival_kmf()
		all_timepoints = np.array(list(timeline_surprob_mapping.keys())).astype(int)
		timepoint_idx_mapping = dict(zip(all_timepoints, range(len(all_timepoints))))

		idx_timepoints = np.array(list(map(lambda x: timepoint_idx_mapping[x], timepoint_arr))).astype(int)
		previdx_timepoints = np.array(list(map(lambda x: max(0, x-1), idx_timepoints))).astype(int)
		prevtimepoint_arr = all_timepoints[previdx_timepoints]

		current_surprob = self.get_cumulative_kmf_surprob(timepoint_arr)/self.get_cumulative_kmf_surprob(prevtimepoint_arr)
		return current_surprob

	def _fit_cph(self, data_df):
		self.cph = lifelines.CoxPHFitter()
		self.cph.fit(data_df, duration_col=self.dur_col, event_col=self.event_col, show_progress=True)
		return self.cph
	
	def __parse_survival_cph(self, data_df):
		if (self.cph is None):
			self._fit_cph(data_df) 
		predict_sur_prob = self.cph.predict_survival_function(data_df)
		cph_survival_table = np.array(predict_sur_prob).T
		all_timepoints = np.array(predict_sur_prob.reset_index().iloc[:, 0]).astype(int)
		return cph_survival_table, all_timepoints

	def get_last_cph_surprob(self, data_df):
		return np.array(self.cph.predict_survival_function(data_df).iloc[-1, :])

	def get_cumulative_cph_surprob(self, data_df):
		timepoint_arr = np.array(data_df[self.dur_col]).astype(int)
		survival_table, all_timepoints = self.__parse_survival_cph(data_df)
		timepoint_idx_mapping = dict(zip(all_timepoints, range(len(all_timepoints))))
		idx_cus = np.array(range(len(data_df))).astype(int)
		idx_dur = np.array(list(map(lambda x: timepoint_idx_mapping[x], np.array(data_df[self.dur_col]).astype(int)))).astype(int)
		return survival_table[idx_cus, idx_dur]

	def get_current_cph_surprob(self, data_df):
		timepoint_arr = np.array(data_df[self.dur_col]).astype(int)
		survival_table, all_timepoints = self.__parse_survival_cph(data_df)
		timepoint_idx_mapping = dict(zip(all_timepoints, range(len(all_timepoints))))
		
		idx_cus = np.array(range(len(data_df))).astype(int)
		idx_dur = np.array(list(map(lambda x: timepoint_idx_mapping[x], np.array(data_df[self.dur_col]).astype(int)))).astype(int)
		prev_dur = np.array(list(map(lambda x: max(0, x-1), idx_dur))).astype(int)
		
		current_surprob = survival_table[idx_cus, idx_dur]/survival_table[idx_cus, prev_dur]
		return current_surprob