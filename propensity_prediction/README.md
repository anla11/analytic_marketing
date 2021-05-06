# Propensity Prediction: Implementation of predicting tasks

## 1. Overview

```
├── README.md                      <- The top-level README for developers using this project.
├── config                         <- Model config + AutoTune, defintion of DataGroup and Data config
│
├── auto_preprocess                <- Cast types, parse config to get data required
│                                     
├── tasks                          <- Predicting tasks
│   └── churn_prediction           
│   └── ...                       
│
├── requirements.txt               <- Make this project installable with `pip install -r`
│
├── setup.py                       <- Make this project installable with `pip install -e`
```
	
## 2. Tasks

- Churn Prediction
[Demo_ChurnPrediction.ipynb](https://github.com/primedata-ai/ds/blob/propensity_prediction/notebooks/Demo_ChurnPrediction_NKI.ipynb)

- Conversion In-Session prediction
[DemoPackage_ConvertingActionPrediction](https://github.com/primedata-ai/ds/blob/propensity_prediction/notebooks/Demo_Conversion_InSession_Prediction.ipynb)

- LTV Prediction
[DemoPackage_LTV_Prediction](https://github.com/primedata-ai/ds/blob/propensity_prediction/notebooks/Demo_LTV_Prediction.ipynb)

## 3. AutoTune

[AutoTune_ChurnPrediction.ipynb](https://github.com/primedata-ai/ds/blob/pipeline/notebooks/AutoTune_ChurnPrediction_pipeline.ipynb)
