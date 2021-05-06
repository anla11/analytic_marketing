import numpy as np
import pandas as pd
import torch
import pyro
from pyro.nn import PyroSample
from pyro.nn import PyroModule
from torch import nn
import pyro.distributions as dist

from pyro.infer import SVI, Trace_ELBO
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal

from model.predicting.abstract_training import Base_TrainingClassifier


class Bayesian_Beta_Regression(PyroModule): 
	def __init__(self, output_dim, input_dim):
		super().__init__()
		self.linear = PyroModule[nn.Linear](input_dim, output_dim)
		# self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, input_dim]).to_event(2))
		# self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([output_dim]).to_event(1))

	def forward(self, x, y=None):
		sigma = pyro.sample("sigma", dist.Uniform(0., 1.))
		log_invodd=self.linear(x).squeeze(-1)
		invodd=torch.exp(log_invodd)
		y_mean = 1/(1+invodd)
		with pyro.plate("data", x.shape[0]):
			obs = pyro.sample("obs", dist.Normal(y_mean, sigma), obs=y)
		return y_mean		
		

class Bayesian_LogisticRegression(PyroModule): 
	def __init__(self, output_dim, input_dim):
		super().__init__()
		self.linear = PyroModule[nn.Linear](input_dim, output_dim)
		# self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, input_dim]).to_event(2))
		# self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([output_dim]).to_event(1))

	def forward(self, x, y=None):
		mean = self.linear(x).squeeze(-1)
		sigmoid = torch.sigmoid(mean)
		with pyro.plate("data", x.shape[0]):
			obs = pyro.sample("obs", dist.Bernoulli(logits=sigmoid), obs=y)
		return sigmoid

class Training_Bayesian_LogisticRegression(Base_TrainingClassifier):
	def __init__(self, output_dim, input_dim, loss_func = torch.nn.MSELoss(reduction='mean'), training_config={'learning_rate':0.001, 'epochs':1000}):
		super().__init__('Bayesian_LogisticRegression', Bayesian_LogisticRegression(output_dim, input_dim), loss_func, training_config)

