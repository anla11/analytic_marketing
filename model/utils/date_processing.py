import numpy as np
import pandas as pd

SYSTEM_FIRST_DATE = pd._libs.tslibs.timestamps.Timestamp("2000-01-01 00:00:00")

def extract_relativedate(date_list):
	if len(date_list) == 0:
		return []
	days = pd.Series(date_list).apply(lambda date: (date - SYSTEM_FIRST_DATE).days)
	return days 

def extract_daterange(relative_days, rangelen = 30, recent_order = True):
	daterange = np.array(relative_days / rangelen).astype(int)
	if recent_order:
		date_max = np.max(daterange)
		daterange = date_max - daterange
	return daterange

def get_first_dates(date_df, cus_id = 'CustomerID', invoicedate_name = 'InvoiceDate'):
	if len(date_df) == 0:
		return None
	active_col = 'Active_%s'%invoicedate_name
	userdate_df = date_df.groupby([cus_id])[invoicedate_name].min().reset_index(name = active_col)
	activedate_mapping = dict(zip(userdate_df[cus_id], userdate_df[active_col]))    
	return activedate_mapping

def get_ages(date_df, cus_id = 'CustomerID', invoicedate_name = 'InvoiceDate', activedate_mapping = None): 
	if len(date_df) == 0:
		return []
	if activedate_mapping is None:
		activedate_mapping = get_first_dates(date_df, cus_id, invoicedate_name)
	activedates = date_df.apply(lambda r: activedate_mapping[r[cus_id]], axis = 1)
	ages = date_df[invoicedate_name] - activedates
	if type(date_df[invoicedate_name].iloc[0]) == np.int64:
		ages = np.array(ages).astype(int)
	elif type(date_df[invoicedate_name].iloc[0]) == pd._libs.tslibs.timestamps.Timestamp:
		ages = ages.apply(lambda r: int(r.days))
	return ages	# count by days