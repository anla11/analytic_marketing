from scipy.stats import pearsonr as ps

def cal_correlation(x, y):
	cor, conf = ps(x, y)
	return cor, conf