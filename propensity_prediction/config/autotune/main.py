from propensity_prediction.config.autotune.config_searcher import ConfigSearcher
from propensity_prediction.config.autotune.tasks import get_task_package, get_pipeline_space

def run(global_config, n_trials = 20):
	task_package = get_task_package(global_config)
	pipeline_space = get_pipeline_space(global_config)
	searcher = ConfigSearcher(task_package, pipeline_space, running_config = {'epochs': 500})
	searcher.search(n_trials = n_trials)
	print ('The best result: ', searcher.best_result)

	if 'pipeline_config_path' in global_config.keys():
		config_path = global_config['pipeline_config_path']
		if (config_path is None) == False:
			searcher.save_bestconfig(config_path)

if __name__=='main':
	json_file = open(sys.argv[1], r) 
	n_trials = open(sys.argv[2], r)
	global_config = json.load(json_file)
	run(global_config, n_trials)


# global_config
# {
# 	'task_name': str, # churn_prediction, conversion_insession_prediction, ltv_prediction
# 	'pipeline_config_path': None or str,
# 	'data_config':
# 	{
# 		'path':
# 		dict_name: dict # DataGroup and description: column_name and types
# 	}
# }