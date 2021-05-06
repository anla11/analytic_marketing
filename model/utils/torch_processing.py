import numpy as np
import torch
from model.utils.np_processing import scale_bypercentile 

def scale_minmax(y):
	y_scale_np, vmin_np, vmax_np = scale_bypercentile(y.data.numpy())
	return torch.from_numpy(y_scale_np).float(), torch.from_numpy(vmin_np).float(), torch.from_numpy(vmax_np).float()

def get_mulogvar(x):
	mu, log_var = torch.mean(x), torch.log(torch.var(x))
	return mu, log_var

def normalize_mulogvar(x, mu, log_var):
	return -0.5 * (log_var + np.log(2*np.pi)) -(0.5 * (1/torch.exp(log_var))* (x-mu)**2)   