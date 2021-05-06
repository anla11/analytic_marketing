'''
	@author: anla-ds
	created date: 8 July, 2020
'''
import numpy as np
import pandas as pd
import torch
import pyro
from pyro.nn import PyroSample
from pyro.nn import PyroModule
from torch import nn
import pyro.distributions as dist
from torch.distributions import constraints

from pyro.infer import SVI, Trace_ELBO
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal

from model.predicting.abstract_training import Base_TrainingRegressor

class Bayesian_LinearRegression(PyroModule): 
	def __init__(self, output_dim, input_dim):
		super().__init__()
		self.linear = PyroModule[nn.Linear](input_dim, output_dim)
		# self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, input_dim]).to_event(2))
		# self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([output_dim]).to_event(1))

	def forward(self, x, y=None):
		mean = self.linear(x).squeeze(-1)
		sigma = pyro.sample("sigma", dist.Uniform(0.1, 1.0))
		with pyro.plate("data", x.shape[0]):
			obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
		return mean

class Training_Bayesian_LinearRegression(Base_TrainingRegressor):
	def __init__(self, output_dim, input_dim, loss_func = torch.nn.MSELoss(reduction='mean'), training_config={'learning_rate':0.001, 'epochs':500}):
		super().__init__('Bayesian_LinearRegression', Bayesian_LinearRegression(output_dim, input_dim), loss_func, training_config)


class Bayesian_PoissonRegression(PyroModule): 
	def __init__(self, output_dim, input_dim):
		super().__init__()
		self.linear = PyroModule[nn.Linear](input_dim, output_dim)

	def forward(self, x, y=None):
		mean = self.linear(x).squeeze(-1)
		sigma = pyro.sample("sigma", dist.Uniform(0.1, 1.0))
		with pyro.plate("data", x.shape[0]):
			obs = pyro.sample("obs", dist.LogNormal(mean, sigma, constraints.greater_than(0.0)), obs=y)
		return mean

class Training_Bayesian_PoissonRegression(Base_TrainingRegressor):
	def __init__(self, output_dim, input_dim, loss_func = torch.nn.MSELoss(reduction='mean'), training_config={'learning_rate':0.001, 'epochs':1000}):
		super().__init__('Bayesian_PoissonRegression', Bayesian_PoissonRegression(output_dim, input_dim), loss_func, training_config)