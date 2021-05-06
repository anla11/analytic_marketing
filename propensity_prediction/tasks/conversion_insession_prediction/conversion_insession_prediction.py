from propensity_prediction.tasks.conversion_insession_prediction.context import Conversion_InSession_Context
from propensity_prediction.tasks.conversion_insession_prediction.conversion_insession_prediction_base import Conversion_InSession_TaskBase

class Conversion_InSession_Task(Conversion_InSession_TaskBase):
	def __init__(self, task_config):
		context = Conversion_InSession_Context(task_config.data_config)
		super().__init__(context, task_config.model_config.get_model_config())

