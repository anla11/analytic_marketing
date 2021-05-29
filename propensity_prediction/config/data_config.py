from propensity_prediction.config.datagroup import USER_PROFILE, PRODUCT_PROFILE, HISTORY, CONSTRAINT
from propensity_prediction.config.base import Base_TaskConfig, Base_DataConfig, HistoryBased_DataConfig, SessionBased_DataConfig

class ChurnPrediction_DataConfig(HistoryBased_DataConfig):
	def __init__(self, data_config):
		super().__init__(data_config)

class LTVPrediction_DataConfig(HistoryBased_DataConfig):
	def __init__(self, data_config):
		super().__init__(data_config)		

class Conversion_Insession_Prediction_DataConfig(SessionBased_DataConfig):
	def __init__(self, data_config):
		super().__init__(data_config)
		