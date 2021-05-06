from propensity_prediction.tasks.LTV_prediction.context import LTVPrediction_Context
from propensity_prediction.tasks.LTV_prediction.LTV_prediction_base import LTVPrediction_TaskBase

class LTVPrediction_Task(LTVPrediction_TaskBase):
	def __init__(self, task_config):
		context = LTVPrediction_Context(task_config.data_config)
		super().__init__(context, task_config.model_config.get_model_config())

