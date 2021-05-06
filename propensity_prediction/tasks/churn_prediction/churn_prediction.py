from propensity_prediction.tasks.churn_prediction.context import ChurnPrediction_Context
from propensity_prediction.tasks.churn_prediction.churnprediction_base import ChurnPrediction_TaskBase

class ChurnPrediction_Task(ChurnPrediction_TaskBase):
	def __init__(self, task_config):
		context = ChurnPrediction_Context(task_config.data_config)
		model_config = task_config.model_config.get_model_config()
		super().__init__(context, model_config)