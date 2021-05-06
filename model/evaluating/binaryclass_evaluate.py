'''
	@author: anla-ds
	created date: 8 July, 2020
'''
import numpy as np
import sklearn.metrics as metrics
from model.evaluating.abstract_evaluation import Abstract_Evaluation

class BinaryClassification_Evaluation(Abstract_Evaluation):
	def __init__(self, binarize_config = {}):
		from model.post_processing.binarize_lib import Binarize
		self.binarize = Binarize(binarize_config)

	def __get_auc(self, obs, prob):
		fpr, tpr, threshold = metrics.roc_curve(obs, prob)
		roc_auc = metrics.auc(fpr, tpr)
		return {'auc': roc_auc}

	def __evaluate_withlabel(self, obs, predict_label):
		n = len(obs)
		n_acc = np.sum(predict_label==obs)
		pos_idx, neg_idx = np.where(obs==1)[0], np.where(obs==0)[0]
		n_tp, n_tn = np.sum(predict_label[pos_idx]==obs[pos_idx]), np.sum(predict_label[neg_idx]==obs[neg_idx])
		n_fp, n_fn = len(pos_idx) - n_tp, len(neg_idx)- n_tn 

		acc =  np.sum(predict_label==obs)*1.0/n
		recall = 1.0*n_tp/(n_tp+n_fn)
		precision = 1.0*n_tp/len(pos_idx) #True positive/Total positive
		trueneg_rate = 1.0*n_tn/len(neg_idx) #True negative/Total negative
		f1_score = 2 * precision * recall / (precision + recall)
		
		return {'accuracy': acc, 'precision': precision, 'recall': recall, 'trueneg_rate': trueneg_rate, 'f1_score': f1_score}

	def __evaluate_with_threshold(self, obs, prob):
		threshold_results = []
		for method in self.binarize.Threshold_Binarize.methods:
			threshold = self.binarize.Threshold_Binarize.get_threshold(prob, method=method)
			result = {'threshold_method':method, 'results':self.__evaluate_withlabel(obs, np.array(prob>threshold).astype(int))}
			threshold_results.append(result)
		return {'method':'threshold', 'list_results':threshold_results}

	def __evaluate_with_gettop(self, obs, prob):
		rate = np.mean(obs)
		n = len(obs)
		fmin, fmax = rate-0.1, rate+0.1
		list_ntops = [max(0, int(fmin*n)), int(n*rate), min(n, int(fmax*n))]
		
		gettop_results = []
		for ntop in list_ntops:
			predict_label = self.binarize.gettop_binarize(prob, ntop, ascending=-1)
			result = {'ntop':ntop, 'results':self.__evaluate_withlabel(obs, predict_label)}
			gettop_results.append(result)
		return {'method':'gettop', 'list_results':gettop_results}

	def _evaluate(self, obs, prob):
		eva_result = self.__get_auc(obs, prob)
		binarize_config = self.binarize.binarize_config
		if 'method' in binarize_config.keys():
			if binarize_config['method'] == 'gettop':
				eva_result.update({'binarize_methods': [self.__evaluate_with_gettop(obs, prob)]})
				return eva_result
			if binarize_config['method'] == 'threshold':
				eva_result.update({'binarize_methods': [self.__evaluate_with_threshold(obs, prob)]})
				return eva_result			
		eva_result.update({'binarize_methods': [self.__evaluate_with_threshold(obs, prob), self.__evaluate_with_gettop(obs, prob)]})
		return eva_result


