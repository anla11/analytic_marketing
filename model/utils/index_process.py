'''
	@author: anla-ds
	created date: 8 July, 2020
'''
import numpy as np
import pandas as pd

def gettopidx_row(x, NTOP):
	res = np.argsort(x)[:NTOP]
	return res

def keeppositive_idx(arr):
	x, y = np.where(arr > 0)
	return x, y

def gettopidx_eachrow_matrix(arr, NTOP, ascending=1):
	'''
		ascending=1: sort ascending then get NTOP
		ascending=-1: sort descending then get NTOP
	'''
	return np.apply_along_axis(gettopidx_row, 1, ascending * arr, NTOP)

def binarize_bygettop(prob, n_top, ascending=1):
	'''
		- Sorted and find NTOP idx of each row
		- Turn NTOP idx to 1 and others to 0
	'''
	n_obj = len(prob)
	select_idx = gettopidx_eachrow_matrix(prob.reshape(1, -1), NTOP=n_top, ascending=ascending)
	# print ("First prob: %.4f, Last prob: %.4f" % (prob[np.array(select_idx)[0, 0]], prob[np.array(select_idx)[0,-1]]))
	binarized_arr = np.zeros((n_obj,))
	binarized_arr[select_idx]=1
	return binarized_arr


def filteridx_matrix(arr, NTOP):
	'''
		- Sorted and limit NTOP idx of each row
		- Remove <= 0 idx of each row
		- Return idx of arr
	'''
	top_idx = gettopidx_eachrow_matrix(arr, NTOP)
	top_value = np.array([arr[i, top_idx[i]] for i in range(len(top_idx))])
	x, y = keeppositive_idx(top_value)
	return x, top_idx[x, y]



class Batch_Data():
	def __init__(self, sample_size, batch_size):
		self.sample_size = sample_size
		self.batch_size = batch_size
		self.n_batch = int((sample_size-1)/batch_size + 1)
		self.cur_batchidx = 0

	def _get_idxrange(self):
		if self.cur_batchidx == self.n_batch:
			return self.n_batch, self.n_batch
		return self.cur_batchidx * self.batch_size, min(self.sample_size, (self.cur_batchidx+1) * self.batch_size -1)

	def is_end(self):
		return (self.cur_batchidx == self.n_batch)

	def enum_batch(self):
		start_idx, end_idx = self._get_idxrange()
		if self.cur_batchidx < self.n_batch:
			self.cur_batchidx +=1
		return start_idx, end_idx	

	def _get_data(self, data):
		start_idx, end_idx = self._get_idxrange()
		__get_data = None
		if type(data) == np.ndarray:
			__get_data = lambda x: x[start_idx: end_idx]
		elif (type(data) == pd.core.frame.DataFrame) or (type(data) == pd.core.series.Series):
			__get_data = lambda x: x.iloc[start_idx: end_idx]
		return __get_data(data)

	def _iter(self, func, data, args = [], outputs = None):
		res = None
		if len(args) == 0:
			res = func(self._get_data(data))
		elif len(args) == 1:
			res = func(self._get_data(data), args[0])
		elif len(args) == 2:
			res = func(self._get_data(data), args[0], args[1])
		if (outputs is None) == False:
			start_idx, end_idx = self._get_idxrange()
			outputs[start_idx: end_idx] = res

	def run(self, func, data, args = [], outputs = None):
		while (self.is_end() == False):
			self._iter(func, data, args = args, outputs = outputs) 
			self.enum_batch()
