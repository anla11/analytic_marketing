import numpy as np

class Abstract_Evaluation():
	def _evaluate(self, obs, prob):
		raise NotImplementedError()

	def _clean(self, obs, prob): # obs and prob must have vector shape (-1)
		valid_idx = np.argwhere(~np.isnan(obs))
		obs, prob = obs[valid_idx].reshape(-1), prob[valid_idx].reshape(-1)
		valid_idx = np.argwhere(~np.isnan(prob))
		obs, prob = obs[valid_idx].reshape(-1), prob[valid_idx].reshape(-1)
		return obs, prob

	def evaluate(self, obs, prob):
		obs, prob = self._clean(obs, prob)
		if len(prob) == 0:
			return {}
		eva = self._evaluate(obs, prob)
		return eva	