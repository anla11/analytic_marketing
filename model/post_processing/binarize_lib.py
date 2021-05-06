'''
	@author: anla-ds
	created date: 8 July, 2020
'''
import numpy as np
from sklearn.cluster import KMeans 

from model.post_processing.image_threshold import ImageThreshold
from model.utils import binarize_bygettop

class Threshold_Binarize(ImageThreshold):
	def __init__(self, INT_RANGE=255):
		'''
			arr: numpy.array() with float elements in range [0, 1]
		'''
		self.INT_RANGE = INT_RANGE
		self.methods = ['constant', 'baseline', 'kmeans', 'otsu', 'yen', 'iso'] 
 
	def kmeans_threshold(self, prob):
		kmeans = KMeans(n_clusters=2, random_state=0).fit(prob.reshape(-1, 1))
		vmin, vmax = np.max(prob[kmeans.labels_==0]), np.min(prob[kmeans.labels_==1])
		return (vmin+vmax)/2

	def get_threshold(self, prob, method='kmeans'):
		if method=='constant':
			return 0.5
		if method=='baseline':
			return 1.0-np.mean(prob)
		if method=='kmeans':
			return self.kmeans_threshold(prob)
		image_threshold = ImageThreshold(prob*self.INT_RANGE)
		if method == 'otsu':
			return np.float(image_threshold.otsu_threshold())/self.INT_RANGE
		if method == 'yen':
			return np.float(image_threshold.yen_threshold())/self.INT_RANGE
		if method == 'iso':
			return np.float(image_threshold.iso_threshold())/self.INT_RANGE
		return None

class Binarize():
	def __init__(self, binarize_config = {'method':'threshold', 'para':{'threshold_method':'kmeans'}}):
		self.Threshold_Binarize = Threshold_Binarize()
		self.binarize_config = {}
		self.update_config('method', binarize_config['method'])
		if binarize_config['method']=='gettop':
			self.update_config('ntop', None)
		elif binarize_config['method']=='threshold':
			self.update_config('threshold_method', 'baseline')
			if 'threshold_method' in binarize_config['para']:
				method = binarize_config['para']['threshold_method']
				if method in self.Threshold_Binarize.methods:
					self.update_config('threshold_method', method)

	def update_config(self, key, value):	
		self.binarize_config[key]=value

	def gettop_binarize(self, prob, ntop, ascending=-1):
		'''
			prob: probabilities array in which all elements are float in range [0, 1]
		'''
		return binarize_bygettop(prob, ntop, ascending)

	def threshold_binarize(self, prob, method, value=None):
		thres = value
		if method in self.Threshold_Binarize.methods:
			thres = self.Threshold_Binarize.get_threshold(prob, method)
		if thres is None: 
			return None
		return np.array(prob>thres).astype(int)

	def binarize(self, prob, ascending=-1):
		if self.binarize_config['method'] == 'gettop':
			return self.gettop_binarize(prob, self.binarize_config['ntop'], ascending)
		if self.binarize_config['method'] == 'threshold':
			return self.threshold_binarize(prob, self.binarize_config['threshold_method'])
		return None