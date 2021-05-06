binarize_space = {
	'suggest': 'choice',
	'para_name': 'method', #name in code
	'space_name': 'binarize.method', #name in optuna
	'space': ['threshold', 'gettop'],
	'child_relationship': 'conditional',

	'sub_space':{
		'threshold': {
			'suggest': 'choice',\
			'para_name': 'threshold_method',\
			'space_name': 'threshold.method',\
			'space': ['constant', 'baseline', 'kmeans', 'otsu', 'yen', 'iso'], \
			},       
		'gettop': {}       
	}, \
}

feature_engineering_space = {
	'para_name': 'method',
	'space': ['pca', 'scoring'],
	'child_relationship': 'joint',
	'sub_space':
	{
		'pca': { 
			'para_name': 'outputdim',
			'space_name': 'pca.outputdim',
			'suggest': 'choice',
			'space': [None, 5, 10, 20, 100],
		},
		'scoring':
		{
			'para_name': 'method',
			'child_relationship': 'multi-para',
			'space': ['impact', 'scale', 'log'],
			'sub_space':
			{
				'impact': {
					'para_name': 'impact',
					'space_name': 'scoring.impact',
					'suggest': 'choice',
					'space': [True, False]
				},
				'scale': {
					'para_name': 'scale',
					'space_name': 'scoring.scale',
					'suggest': 'choice',
					'space': [True, False]
				},
				'log': {
					'para_name': 'log',
					'space_name': 'scoring.log',
					'suggest': 'choice',
					'space': [True, False]
				},
			}
		}
}}

classification_model_space = {
	'para_name': 'model_name',
	'child_relationship': 'joint',
	'space': ['LogisticRegression', 'Bayesian_LogisticRegression'],
	'sub_space':
	{
		'LogisticRegression': {
			'para_name': 'learning_rate',
			'space_name': 'LogisticRegression.learning_rate',
			'suggest': 'float',
			'space': {'low': 0.0005, 'high': 0.01, 'log':False, 'step': 0.0005}
		},
		'Bayesian_LogisticRegression': {
			'para_name': 'learning_rate',
			'space_name': 'Bayesian_LogisticRegression.learning_rate',
			'suggest': 'float',
			'space': {'low': 0.0005, 'high': 0.01, 'log':False, 'step': 0.0005}
		}
}}

classification_pipeline_space = {
	'para_name': 'classification_pipeline',
	'space': ['binarize_config', 'feature_engineering', 'model_config'],
	'child_relationship': 'joint',
	'sub_space':    
	{
		'binarize_config': binarize_space,
		'feature_engineering': feature_engineering_space, 
		'model_config': classification_model_space
	}
}


regression_model_space = {
	'para_name': 'model_name',
	'child_relationship': 'joint',
	'space': ['LinearRegression', 'PoissonRegression', 'Bayesian_LinearRegression', 'Bayesian_PoissonRegression'],
	'sub_space':
	{
		'LinearRegression': {
			'para_name': 'learning_rate',
			'space_name': 'LinearRegression.learning_rate',
			'suggest': 'float',
			'space': {'low': 0.001, 'high': 0.1, 'log':False, 'step': 0.001}
		},
		'PoissonRegression': {
			'para_name': 'learning_rate',
			'space_name': 'PoissonRegression.learning_rate',
			'suggest': 'float',
			'space': {'low': 0.001, 'high': 0.1, 'log':False, 'step': 0.001}
		},
		'Bayesian_LinearRegression': {
			'para_name': 'learning_rate',
			'space_name': 'Bayesian_LinearRegression.learning_rate',
			'suggest': 'float',
			'space': {'low': 0.001, 'high': 0.1, 'log':False, 'step': 0.001}
		},
		'Bayesian_PoissonRegression': {
			'para_name': 'learning_rate',
			'space_name': 'Bayesian_PoissonRegression.learning_rate',
			'suggest': 'float',
			'space': {'low': 0.001, 'high': 0.1, 'log':False, 'step': 0.001}
		},
}}

regression_pipeline_space = {
	'para_name': 'regression_pipeline',
	'space': ['feature_engineering', 'model_config'],
	'child_relationship': 'joint',
	'sub_space':    
	{
		'feature_engineering': feature_engineering_space, 
		'model_config': regression_model_space
	}
}