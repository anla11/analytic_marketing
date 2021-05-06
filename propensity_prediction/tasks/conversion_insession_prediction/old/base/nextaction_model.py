import numpy as np
import pandas as pd 
from propensity_prediction.tasks.converting_action_prediction.base.context import NextAction_Context
from propensity_prediction.tasks.base import Task_Base
from propensity_prediction.tasks.converting_action_prediction.base.nextaction_base \
	import NextAction_BinaryClass_Base, NextAction_MultiClass_Base, NextAction_Ensemble_Base

class NextAction_BinaryClasses(NextAction_Ensemble_Base):
	def __init__(self, context_base, model_config_base= {'dropna_preprocess':True, 'binarize_config':{'method':'threshold', 'thres_type':None}}):
		super().__init__(context_base, model_config={'model_name':'NextAction_BinaryClass_Ensemble', 'multi_class':False})
		self.list_models = []
		for order_action in self.context.order_actions:
			new_context = self.context.copy()
			new_context.order_actions = [order_action]
			model_config = model_config_base.copy()
			model_config['model_name'] = 'NextAction_BinaryClass_Ensemble'           
			model_config['multi_class'] = False     
			model = NextAction_BinaryClass_Base(new_context, model_config)
			self.list_models.append(model)   
            
class NextAction_MultiClass(NextAction_Ensemble_Base):
	def __init__(self, context, model_config_base=None):
		super().__init__(context, model_config={'model_name':'NextAction_MultiClass_Ensemble', 'multi_class':True})
		self.list_models = [NextAction_MultiClass_Base(context)]

class NextAction_Task(Task_Base, NextAction_Ensemble_Base):
	def __init__(self, context, model_config):
		self.model, self.model_name = None, model_config['model_name']#'NextAction'         
		self.list_models = []
		for sub_model_config in model_config['methods']: 
			if sub_model_config['model_name'] == "BinaryClasses":
				model = NextAction_BinaryClasses(context, sub_model_config['model_config'])
				self.list_models.append(model)
			if sub_model_config['model_name'] == "MultiClass":
				model = NextAction_MultiClass(context)
				self.list_models.append(model)
					