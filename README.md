# Pipeline for analtyic marketing tasks

In online websites or e-commerce services, the activities of users are informative to gain insights about how to improve marketing profits. They are extracted and organized as features of user behaviours. Then they are used to predict propensity values, which reflect user tendencies. Furthermore, values of users are evaluated (e.g. VIP, premium, potential) in terms of marketing profits (KPIs), and these features are useful inputs to discover underlying user-behaviour patterns that impact user values. These interesting ideas are widely applied nowaday, also introduced in the book [Introduction to Algorithmic Marketing](https://www.algorithmicmarketingbook.com). 

However, [Exploratory Data Analysis](https://en.wikipedia.org/wiki/Exploratory_data_analysis) requires many manual efforts of data analysts. Moreover, to find underlying patterns, current methods of Segment Analytics, such as clustering, are insufficiently supervised by expert knowledge. For examples, Marketing experts know that VIP users and normal users have differences in some behaviors, and we wish to obtain groups of users implying that insight. But traditional clustering methods barely return desired results, require careful discovery by experts, analysts, and deployers.

This project proposes a framework to show that these 2 difficult problems of Algorithmic Marketing are possible to solve. It provides demonstrations of: 
- Automating [Exploratory Data Analysis](https://en.wikipedia.org/wiki/Exploratory_data_analysis)
- Automating Segment Analytics: Generating segments of users (called clusters in Computer Science) in favour of KPIs (total money, number of products). Segments of users serve to discover user-behaviour patterns related to KPIs tendencies. In particular, a segment is a group of users such that: 
	- (1) all of its users have relatively the same behaviours and KPI values
	- (2) the average KPI values are different to other groups.
In addition, predicting propensity values (churn rate, conversion rate, lifetime value) are included.

The project is deployed as a standard Machine Learning pipeline: data preprocessing, feature engineering, modeling, and evaluating. Automatic EDA has not been fully developed, but its basic ideas is provided in [model/analytics](model/analytics). 
Automatically tuning parameters is included in this project. 
For more details of data format and pipeline usage, see [examples](examples) and [notebooks](notebooks).

## 1. Overview

```
├── README.md                      <- The top-level README for developers using this project.
│
├── model			   <- ML Standard pipeline: data preprocessing, feature engineering, modeling, evaluating
│
├── propensity_prediction          <- Package for predicting tasks 
│
├── segment			   <- Package for automatically generating segments 
│
├── notebooks                      <- Jupyter notebooks. Naming convention is a number (for ordering),
│                                     the creator's initials, and a short `-` delimited description, e.g.
│                                     `1.0-jqp-initial-data-exploration`.
│
│
├── reports                        <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                    <- Generated graphics and figures to be used in reporting
│   └── docs                       <- A default Sphinx project; see sphinx-doc.org for details
│   └── references                 <- Data dictionaries, manuals, and all other explanatory materials.
│
├── requirements.txt               <- The requirements file for reproducing the analysis environment, e.g.
│                                     generated with `pip freeze > requirements.txt`
│
├── setup.py                       <- Make this project pip installable with `pip install -e`
```

## 2. Tasks

- Churn Prediction
[Demo_ChurnPrediction.ipynb](notebooks/Demo_ChurnPrediction_NKI.ipynb)

- Conversion In-Session prediction
[DemoPackage_ConvertingActionPrediction](notebooks/Demo_Conversion_InSession_Prediction.ipynb)

- LTV Prediction
[DemoPackage_LTV_Prediction](notebooks/Demo_LTV_Prediction.ipynb)

- Segment
[notebooks/Segment Demo.ipynb](notebooks/Segment%20Demo.ipynb)

## 3. AutoTune for propensity prediction

[AutoTune_ChurnPrediction.ipynb](notebooks/AutoTune_ChurnPrediction_pipeline.ipynb)

## 4. Set-up

### Requirements
> python 3.7+


### Pull image and create container

	<workspace_host> = "D:\\Workspace\\git"
	<file_sharing> = "D:\\Workspace\\docker\\file_sharing"
	<dataset_dir> = "D:\\Workspace\\docker\\dataset"
		
	docker pull pytorch/pytorch
	docker create -it --name <container_name> -v <workspace_host>:/src -v <file_sharing>:/home/share -v <dataset_dir>:/home/dataset -p 8888:8888 -t pytorch/pytorch
	docker start <container_name>
	docker exec -it <container_name> /bin/bash

### Install base packages
	
	<python_path> = /opt/conda/bin/python

	apt-get update
	apt-get upgrade
	apt-get install g++
	apt-get install -y swig curl
	pip install pipenv ipykernel notebook

### init environment

	cd to_project
	pipenv --python <python_path>

	pipenv shell
	pip install -r <requirement_file>
	pip freeze >> full_requirement.txt

	pipenv install ipykernel
	python -m ipykernel install --name=<name_package>
	exit

### open jupyter notebook 

	cd /src
	jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root

