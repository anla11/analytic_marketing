import numpy as np

def scale_bypercentile(y, scale_values = None):
	# vmin, vmax = np.min(y, axis=0), np.max(y, axis=0)
	vmin, vmax = None, None
	if scale_values is None:
		vmin, vmax = np.percentile(y, q=2.5, axis=0), np.percentile(y, q=97.5, axis=0)
	else:
		vmin, vmax = scale_values
		
	y_scale = (y-vmin)/(vmax - vmin)
	y_scale[y_scale>1] = 1
	y_scale[y_scale<0] = 0
	return np.array(y_scale), np.array(vmin), np.array(vmax)

def scale_signunit(x): # return distribution with loc=0 and fluctuation_range = 1
	vmin, vmean, vmax = np.min(x), np.mean(x), np.max(x)
	scale = max(vmean-vmin, vmax-vmean)
	return (x-vmean)/scale

def scale_minmax(x):
	vmin, vmax = np.min(x), np.max(x)
	return (x- vmin)/(vmax-vmin)