from propensity_prediction.config.datagroup import USER_PROFILE, PRODUCT_PROFILE, HISTORY, CONSTRAINT
from propensity_prediction.config.base import Base_TaskConfig, Base_ModelConfig

class ChurnPrediction_ModelConfig(Base_ModelConfig):
	def __init__(self, pipeline_config_path = None):
		from propensity_prediction.config.detail_model_config.churn_prediction import churn_pipeline_config as default_config
		super().__init__('Ensemble', default_config, pipeline_config_path, \
											list_configs = ['binarize_config', 'feature_engineering', 'model_config'])


class Conversion_InSession_Prediction_ModelConfig(Base_ModelConfig):
	def __init__(self, pipeline_config_path = None):
		from propensity_prediction.config.detail_model_config.conversion_insession_prediction import conversion_insession_pipeline_config as default_config
		super().__init__('Ensemble', default_config, pipeline_config_path, \
											list_configs = ['binarize_config', 'feature_engineering', 'model_config'])


class LTVPrediction_ModelConfig(Base_ModelConfig):
	def __init__(self, pipeline_config_path = None):
		from propensity_prediction.config.detail_model_config.ltv_prediction import ltv_prediction_pipeline_config as default_config
		super().__init__('Ensemble', default_config, pipeline_config_path,\
											list_configs = ['feature_engineering', 'model_config'])
