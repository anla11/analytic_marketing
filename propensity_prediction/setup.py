#!/usr/bin/env python
from setuptools import setup, find_packages

requires = [
	"torch==1.7.0",
	"pyro-ppl==1.4.0",
	"lifelines==0.24.16",
	"scikit-learn==0.22.2",
	"pandas==1.0.5",
	"seaborn==0.10.1",
	"xlrd==1.2.0",
	"optuna==2.7.0"
]

# conflicts resolution, see https://gitlab.com/meltano/meltano/issues/193
conflicts = []

dev_requires = [

]

infra_requires = []

setup(
	name="propensity_prediction",
	version="0.1.0",
	author="An La",
	author_email="an.la@primedata.ai",
	description="Propensity Prediction",
	long_description="Predictive tasks for marketing problems",
	long_description_content_type="text/markdown",
	url="https://github.com/primedata-ai/ds_experience",
	packages=find_packages(),
	include_package_data=True,
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	setup_requires=["pytest-runner"],
	tests_require=dev_requires,
	# run `make requirements.txt` after editing
	install_requires=[*conflicts, *requires],
	extras_require={"dev": dev_requires, "infra": infra_requires},
)
