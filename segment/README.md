# Segment Analytics

## 1. Overview

```
├── README.md                      <- The top-level README for developers using this project.
│
├── Custom Segment                 <- generate segments by KPIs
│   └── Custom_Segment.py           
│
├── Generic Segment                
│   └── Generic_Segment.py         <- generate segments in general 
│   └── Segment_Analytics.py       <- call Generic_Segment and Custom_Segment
│
├── Segment_Data.py                <- Reading data and become input of Segment_Analytics
│
├── visualize_segmentana.py        <- For visualization
│
├── requirements.txt               <- Make this project installable with `pip install -r`
│
├── setup.py                       <- Make this project installable with `pip install -e`
```
	
## 2. Usage

	```
	from model import compute_clusters, visualize_clusters
	from segment.Generic_Segment.Segment_Analytics import Segment_Analytics
	```
	+ generate_CustomCluster(): generate clusters by KPIs
	+ generate_GenericSegment(): generate segments in general

## 3. Demo
[Demo_Segment.ipynb](https://github.com/primedata-ai/ds/blob/segment/notebooks/Segment%20Demo.ipynb)
