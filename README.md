# Pipeline for propensity_prediction

## 1. Overview
Experiences of Data Science Team

```
├── README.md                      <- The top-level README for developers using this project.
├── propensity_prediction          <- Propensity package: data preprocessing, feature engineering, modeling, evaluating
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

- Pipeline [Demo of Pipeline.ipynb](notebooks/Demo_New_Pipeline.ipynb)

- Churn Prediction
[Demo_ChurnPrediction.ipynb](notebooks/Demo_ChurnPrediction_NKI.ipynb)

- Conversion In-Session prediction
[DemoPackage_ConvertingActionPrediction](notebooks/Demo_Conversion_InSession_Prediction.ipynb)

- LTV Prediction
[DemoPackage_LTV_Prediction](notebooks/Demo_LTV_Prediction.ipynb)

## 3. AutoTune

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

