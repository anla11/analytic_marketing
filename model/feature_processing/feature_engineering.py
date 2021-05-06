'''
	@author: anla-ds
	created date: 8 July, 2020
'''
import numpy as np
from model.utils import scale_bypercentile, scale_minmax

def get_values(x = {}, key = ''):
	if x is None:
		return None
	if key in x.keys():
		return x[key]
	return None	

class Abstract_FeatureEngineering():
	def __init__(self, name, args={}):
		self.name = name
		self.args = args
	
	def process(self, x, y, constants):
		pass

	def get_arg(self, key):
		if key in self.args.keys():
			return self.args[key]
		return None
	
	def _get_constant(self, constant, key):
		return get_values(constant, key)

class Scoring_Feature(Abstract_FeatureEngineering):
	def __init__(self, args = {'impact':True, 'scale':True, 'log':True}):
		super().__init__('scoring', args)

	def _get_impact(self, x, y, constant):
		def _cal_slope(x, y):
			from scipy.stats import linregress
			slope, intercept, r_value, p_value, std_err = linregress(x, y)
			return slope
		
		impact = 1
		if (constant is None) == False:
			impact = self._get_constant(constant, 'impact')
		elif (y is None) == False:
			impact = (_cal_slope(x, y) > 0) *2 -1
		return x*impact, impact

	def _get_log(self, x, y, constant):
		flat = False
		if (constant is None) == False:
			flat = self._get_constant(constant, 'log_flat')
		elif (y is None) == False:
			import pandas as pd
			threshold = 2.0
			skew = pd.Series(x).skew()
			if (np.min(x) > 0) and ((skew < -threshold) or (skew > threshold)):
				flat = True
		if flat:		
			new_x = scale_minmax(np.log(x + 1e-10))
			return new_x, True
		return x, False

	def _get_scaled(self, x, y, constant):
		scale_values = None
		if (constant is None) == False:
			scale_values = self._get_constant(constant, 'scaled')
		new_x, xmin, xmax = scale_bypercentile(x, scale_values)
		return new_x, (xmin, xmax)

	def _get_score(self, x, y, constant):
		new_x = x
		new_constant = {}
		
		if self.args['impact']:
			new_x, impact = self._get_impact(new_x, y, constant)
			new_constant['impact'] = impact
		if self.args['scale']:
			new_x, scaled = self._get_scaled(new_x, y, constant)
			new_constant['scaled'] = scaled
		if self.args['log']:
			new_x, log_flat = self._get_log(new_x, y, constant)
			new_constant['log_flat'] = log_flat
		return new_x, y, new_constant

	def process(self, X, y, constants):
		if constants is None:
			constants = {}
			for c in range(X.shape[1]):
				constants[c] = None
		new_X = np.zeros(X.shape)
		for c in range(X.shape[1]):
			new_X[:, c], _, constants[c] = self._get_score(X[:, c], y, constants[c])
		return new_X, y, constants

class Principle_Component_Analysis(Abstract_FeatureEngineering):
	def __init__(self, args):
		super().__init__('pca', args) 
		self.output_dim = self.get_arg('output_dim')
		self.rowvar = self.get_arg('output_dim')
		self.transform_matrix, self.variance = None, None

	def __find_pca(self, x, rowvar=False):
		cov_matrix = np.cov(x, rowvar=rowvar)
		eigen_value, eigen_vector = np.linalg.eig(cov_matrix)
		variation = (abs(eigen_value)*1.0/np.sum(abs(eigen_value)))
		self.transform_matrix = eigen_vector[:,:self.output_dim] # new dimensions
		self.variation=variation[:self.output_dim]
		new_x = np.matmul(x, self.transform_matrix)  #points on new dimensions
		return new_x

	def process(self, X, y=None, constants=None):
		if self.output_dim is None:
			return X, y, constants
		new_X = None
		if self.transform_matrix is None:
			new_X = self.__find_pca(X)
		if new_X is None:
			new_X = np.matmul(X, self.transform_matrix)
		return new_X, y, constants

	def decode_lineartransform(self, W):
		'''
			new_x = x * tranform_matrix
			loss = loss_func(new_x * W + b) = loss_func(x * transform_matrix * W + b)
			new_W = transform_matrix * W
		''' 
		new_W = self.transform_matrix * W
		return new_W

class Feature_Engineering():
	def __init__(self, list_config = []):
		self.list_config = list_config
		self.fe_list = []
		self.output_dim = None
		for config in list_config:
			method, args = get_values(config, 'method'), get_values(config, 'para')
			fe = None
			if method == 'pca':
				fe = Principle_Component_Analysis(config['para'])
				output_dim = fe.get_arg('output_dim')
				if (output_dim is None) == False:
					self.output_dim = output_dim
			if method == 'scoring':
				fe = Scoring_Feature(args)
			self.fe_list.append(fe)
	
	def process(self, X, y, constants):
		new_X = X
		if (constants is None) == True:
			constants = {}
			for fe in self.fe_list:
				constants[fe.name] = None

		for fe in self.fe_list:
			new_X, _, constants[fe.name] = fe.process(new_X, y, constants[fe.name])
		# print (new_X)
		return new_X, y, constants