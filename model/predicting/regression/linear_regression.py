import torch
from torch import nn
from model.predicting.abstract_training import Base_TrainingRegressor

class LinearRegression(torch.nn.Module):
	def __init__(self, output_shape, input_shape):
		super(LinearRegression, self).__init__()
		self.linear = torch.nn.Linear(input_shape, output_shape, bias=False)
	def forward(self, x):
		mean = self.linear(x)
		return mean

class Training_LinearRegression(Base_TrainingRegressor):
	def __init__(self, output_shape, input_shape, loss_func=torch.nn.MSELoss(), training_config={'learning_rate':0.001, 'epochs':500}):
		super().__init__('LinearRegression', LinearRegression(output_shape, input_shape), loss_func, training_config)
		self.early_stopping = None


class PoissonRegression(torch.nn.Module):
	def __init__(self, output_shape, input_shape):
		super(PoissonRegression, self).__init__()
		self.linear = torch.nn.Linear(input_shape, output_shape, bias=False)
		# self.softplus = torch.nn.Softplus()
	def forward(self, x):
		t = torch.sigmoid(x)
		h = self.linear(t)
		mean = torch.exp(h)
		return mean
class Training_PoissonRegression(Base_TrainingRegressor):
	def __init__(self, output_shape, input_shape, loss_func=torch.nn.MSELoss(), training_config={'learning_rate':0.001, 'epochs':500}):
		super().__init__('PoissonRegression', PoissonRegression(output_shape, input_shape), loss_func, training_config)
		self.early_stopping = None