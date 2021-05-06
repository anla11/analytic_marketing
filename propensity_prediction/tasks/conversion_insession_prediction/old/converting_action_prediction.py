from propensity_prediction.tasks.converting_action_prediction.base.context import NextAction_InSession_Context
from propensity_prediction.tasks.converting_action_prediction.base.nextaction_model import NextAction_Task  

class NextAction_InSession_Task(NextAction_Task):
	def __init__(self, task_config):
		context = NextAction_InSession_Context(task_config.data_config)
		super().__init__(context, task_config.model_config.get_model_config())
