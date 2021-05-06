conversion_insession_pipeline_config = {\
	'binarize_config': 
		{'method': 'threshold',	'para': {'threshold_method': 'kmeans'}},   
	'feature_engineering':  
		[	{'method': 'pca', 'para': {'outputdim': None}},
			{'method': 'scoring', 'para': {'impact': False, 'log': False, 'scale': False}}
		],
	'model_config': 
		[	{'method': 'LogisticRegression', 'para': {'epochs': 1000, 'learning_rate': 0.001}},
			{'method': 'Bayesian_LogisticRegression', 'para': {'epochs': 1000, 'learning_rate': 0.001}}
		]
}