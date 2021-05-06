'''
	@author: anla-ds
	created date: 8 July, 2020
'''
import torch
from torch import nn
from model.predicting.abstract_training import Base_TrainingClassifier

class LogisticRegression(nn.Module):
	def __init__(self, output_dim, input_dim):
		super(LogisticRegression, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim)

	def forward(self, X):
		mean = self.linear(X).squeeze(-1)
		y_hat = torch.sigmoid(mean)
		return y_hat

class Training_LogisticRegression(Base_TrainingClassifier):
	def __init__(self, output_dim, input_dim, loss_func=torch.nn.BCELoss(), training_config={'learning_rate':0.001, 'epochs':1000}):
		super().__init__('LogisticRegression', LogisticRegression(output_dim, input_dim), loss_func, training_config)