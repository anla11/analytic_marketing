'''
	@author: anla-ds
	created date: 8 July, 2020
'''
import numpy as np
import torch
from model.utils import Batch_Data, scale_minmax

class Abstract_Training():
	'''
		Abstract for Classifiers: receive X as features and learn how to classify with label y
		+ init: model_name, training_config
		+ training: call fit(X, y)
		+ predicting labels: call predict(X)
		+ predicting probabilities: call predict_proba(X)
		+ package for saving and loading: package_for_save, load_from_savepackage
	'''
	def __init__(self, model_name, training_config={'learning_rate':0.01, 'epochs':500}):
		self.model = None
		self.name = model_name
		self.learning_rate = training_config['learning_rate']
		self.epochs = training_config['epochs']

	def fit(self, X, y, training_batchsize=100):
		raise NotImplementedError()
	
	def predict_proba(self, np_X, predicting_batchsize=100):
		raise NotImplementedError()

	def predict(self, X, predicting_batchsize=100):
		raise NotImplementedError()

	def package_for_save(self, X):
		raise NotImplementedError()		

	def load_from_savepackage(self, X):
		raise NotImplementedError()		

class Base_Training(Abstract_Training):
	def __init__(self, model_name, model, loss_func, training_config={'learning_rate':0.001, 'epochs':1000}):
		super().__init__(model_name, training_config)
		self.model = model
		self.loss_func = loss_func
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
		self.early_stopping, self.best_loss, self.max_patient = None, None, self.epochs/10

	def _totensor(self, np_floatarr):
		return torch.tensor(np_floatarr).float()
		
	def _fit(self, np_X, np_y): 
		X, y = self._totensor(np_X), self._totensor(np_y)
		show_process, split_epoch = True, self.epochs/5
		self.early_stopping, self.best_loss = None, None
		patient = self.max_patient
		for epoch in range(self.epochs):
			y_pred = self.model(X).squeeze(-1)
			loss = self.loss_func(y_pred, y)
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()
			if (self.best_loss is None) or (loss < self.best_loss):
				if (self.best_loss is None) == False:
					if loss < self.best_loss * 0.975: #improve 2.5% comparing to the best
						patient = self.max_patient
					else:
						patient -= 1
				self.best_loss = loss
			else:
				patient -= 1
			if (show_process) and (epoch==0 or epoch % split_epoch==split_epoch-1):
				print("	+ [iteration %04d] Loss: %.4f, Best loss: %.4f" % (epoch + 1, loss / len(X), self.best_loss / len(X)))
			if (loss < 0) or (patient <= 0):
				self.early_stopping = epoch
				print("	+ [iteration %04d] Loss: %.4f, Best loss: %.4f" % (epoch + 1, loss / len(X), self.best_loss / len(X)))
				print ('    -> Early stop at epoch', epoch + 1)
				break
		return self    

	def fit(self, np_X, np_y, training_batchsize =100):
		batch_data_obj = Batch_Data(np_X.shape[0], training_batchsize)
		while (batch_data_obj.is_end() == False):
			print ('    Training with minibacth ', batch_data_obj.cur_batchidx)
			start_idx, end_idx = batch_data_obj.enum_batch()
			self._fit(np_X[start_idx:end_idx, :], np_y[start_idx:end_idx])

	def package_for_save(self):
		return self.model.state_dict()

	def load_from_savepackage(self, state_dict, mode = 'training'):
		self.model.load_state_dict(state_dict)
		if mode == 'testing':
			self.model.eval()
		return self		

class Base_TrainingClassifier(Base_Training):
	def __init__(self, model_name, model, loss_func, training_config={'learning_rate':0.001, 'epochs':500}):
		super().__init__(model_name, model, loss_func, training_config) 
		self.thres = 0.5  

	def _predict_proba(self, np_X):
		X = self._totensor(np_X)
		y_pred = self.model(X).squeeze()
		prob = scale_minmax(np.array(y_pred.squeeze().data.numpy())).reshape(-1, 1)
		return np.hstack([1-prob, prob])

	def predict_proba(self, np_X, predicting_batchsize = 100):
		batch_data_obj = Batch_Data(np_X.shape[0], predicting_batchsize)
		prob = np.zeros((np_X.shape[0], 2))
		batch_data_obj.run(self._predict_proba, data = np_X, outputs = prob)	
		return prob

	def predict(self, np_X, predicting_batchsize = 100):
		predict = np.array(self.predict_proba(np_X, predicting_batchsize = predicting_batchsize)[:, 1]>=self.thres).reshape(-1, 1)
		return predict 			      

class Base_TrainingRegressor(Base_Training):
	'''
		Abstract for Regressor: receive X as features and learn how to classify with label y
		+ init: model_name, training_config
		+ training: call fit(X, y)
		+ predicting labels: call predict(X)
		+ predicting probabilities: call predict_proba(X)
	'''
	def __init__(self, model_name, model, loss_func, training_config={'learning_rate':0.001, 'epochs':500}):
		super().__init__(model_name, model, loss_func, training_config)

	def _predict_proba(self, np_X):
		X = self._totensor(np_X)
		y_pred = self.model(X).squeeze()
		prob = np.array(y_pred.squeeze().data.numpy())
		return prob.reshape(-1, 1)

	def predict_proba(self, np_X, predicting_batchsize = 100):
		batch_data_obj = Batch_Data(len(np_X), predicting_batchsize)
		prob = np.zeros((len(np_X), 1))
		batch_data_obj.run(self._predict_proba, data = np_X, outputs = prob)	
		return prob

	def predict(self, np_X, predicting_batchsize = 100):
		predict = self.predict_proba(np_X, predicting_batchsize = predicting_batchsize) 
		return predict