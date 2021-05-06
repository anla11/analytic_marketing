ltv_prediction_pipeline_config = {\
	'feature_engineering':  
		[	{'method': 'pca', 'para': {'outputdim': None}},
			{'method': 'scoring', 'para': {'impact': True, 'log': True, 'scale': True}}
		],
	'model_config': 
		[	{'method': 'LinearRegression', 'para': {'epochs': 1000, 'learning_rate': 0.1}},
			{'method': 'PoissonRegression', 'para': {'epochs': 1000, 'learning_rate': 0.1}},
			{'method': 'Bayesian_LinearRegression', 'para': {'epochs': 1000, 'learning_rate': 0.1}},
			{'method': 'Bayesian_PoissonRegression', 'para': {'epochs': 1000, 'learning_rate': 0.1}}
		]
}
