import json

def save_json(data, path):
	f = open(path,"w")
	json.dump(data,f)
	f.close()

def load_json(path):
	f = open(path,"r")
	data = json.load(f)
	f.close()
	return data 

import urllib.request
import pickle

def save_picklejson(data, path):
	f = open(path,"wb")
	pickle.dump(data,f)
	f.close()

def load_picklejson(path):
	f = open(path,"rb")
	data = pickle.load(f)
	f.close()
	return data 
	
def load_picklejson_url(url):
	data = pickle.load(urllib.request.urlopen(url))
	return data