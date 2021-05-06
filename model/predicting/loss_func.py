import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
from model.utils import normalize_mulogvar, get_mulogvar

class KLDiv_Loss(torch.nn.modules.loss._Loss):
	def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', log_target: bool = False):
		super(KLDiv_Loss, self).__init__(size_average, reduce, reduction)
		self.log_target = log_target

	def forward(self, input, target):
		mu_p, logvar_p = get_mulogvar(target)
		log_p = normalize_mulogvar(target, mu_p, logvar_p)
		mu_q, logvar_q = get_mulogvar(target)
		log_q = normalize_mulogvar(input, mu_q, logvar_q)         
		return torch.nn.functional.kl_div(log_q, log_p, reduction=self.reduction, log_target=self.log_target)

class ELBO_Loss(torch.nn.modules.loss._Loss):
	def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', log_target: bool = False):
		super(ELBO_Loss, self).__init__(size_average, reduce, reduction)
		self.log_target = log_target
		# self.KL_Div = KLDiv_Loss(size_average, reduce, reductio, log_target = True)

	def forward(self, input, target):
		mu_p, logvar_p = get_mulogvar(target)
		log_p = normalize_mulogvar(target, mu_p, logvar_p)
		mu_q, logvar_q = get_mulogvar(target)
		log_q = normalize_mulogvar(input, mu_q, logvar_q)         
		kl_div = torch.nn.functional.kl_div(log_q, log_p, reduction=self.reduction, log_target=self.log_target)
		reconstruct_loss = normalize_mulogvar(target, mu_q, logvar_q)
		if self.reduction == 'mean':
			reconstruct_loss = torch.mean(log_p)
		else:
			reconstruct_loss = torch.sum(log_p)
		elbo = reconstruct_loss - kl_div
		return -elbo

class Combining_Loss(torch.nn.modules.loss._Loss):
	def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', log_target: bool = False):
		super(Combining_Loss, self).__init__(size_average, reduce, reduction)
		self.log_target = log_target

	def forward(self, input, target):
		mu_p, logvar_p = get_mulogvar(target)
		log_p = normalize_mulogvar(target, mu_p, logvar_p)
		mu_q, logvar_q = get_mulogvar(target)
		log_q = normalize_mulogvar(input, mu_q, logvar_q)           

		kl_loss = torch.nn.functional.kl_div(log_q, log_p, reduction=self.reduction, log_target=self.log_target)
		reconstruct_loss = torch.nn.functional.mse_loss(input, target, reduction=self.reduction)

		loss = -(reconstruct_loss - kl_loss)
		return loss  

class KLandMSE_Loss(torch.nn.modules.loss._Loss):
	def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', log_target: bool = False, alpha = 0.8):
		super(KLandMSE_Loss, self).__init__(size_average, reduce, reduction)
		self.alpha = alpha
		self.log_target = log_target

	def forward(self, input, target):
		mu_p, logvar_p = get_mulogvar(target)
		log_p = normalize_mulogvar(target, mu_p, logvar_p)
		mu_q, logvar_q = get_mulogvar(target)
		log_q = normalize_mulogvar(input, mu_q, logvar_q)           

		kl_loss = torch.nn.functional.kl_div(log_q, log_p, reduction=self.reduction, log_target=self.log_target)
		mse_loss = torch.nn.functional.mse_loss(input, target, reduction=self.reduction)

		loss = (1-self.alpha)*mse_loss + self.alpha * kl_loss
		return loss  

class ELBOandMSE_Loss(torch.nn.modules.loss._Loss):
	def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', log_target: bool = False):
		super(ELBOandMSE_Loss, self).__init__(size_average, reduce, reduction)
		self.log_target = log_target

	def forward(self, input, target):
		mu_p, logvar_p = get_mulogvar(target)
		log_p = normalize_mulogvar(target, mu_p, logvar_p)
		mu_q, logvar_q = get_mulogvar(target)
		log_q = normalize_mulogvar(input, mu_q, logvar_q)           

		kl_loss = torch.nn.functional.kl_div(log_q, log_p, reduction=self.reduction, log_target=self.log_target)
		reconstruct_loss = normalize_mulogvar(target, mu_q, logvar_q)
		if self.reduction == 'mean':
			reconstruct_loss = torch.mean(log_p)
		else:
			reconstruct_loss = torch.sum(log_p)
		elbo_loss = reconstruct_loss - kl_loss
		mse_loss = torch.nn.functional.mse_loss(input, target, reduction=self.reduction)

		loss = 0.5* (mse_loss + elbo_loss)
		return loss