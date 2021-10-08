########################################
# ML Project Setup
########################################

SHELL := $(shell which bash)

.PHONY: help
help:
	@echo -e ""


# ##############################
# Python - local venv
# ##############################
.PHONY: venv install run test configure pytest clean

model = "xgboost"
deps = requirements.txt
run_script = run.py

venv_name = venv
local_python = $(shell which python3)
python = $(venv_name)/bin/python3
pip = $(venv_name)/bin/pip3

venv: venv/bin/activate
venv/bin/activate: $(deps) FORCE
	@echo "Setup python venv"
	if [ ! -d $(venv_name) ]; then \
		$(local_python) -m venv $(venv_name); \
		$(python) -m pip install -r $(deps); \
		@printf "export LOCAL_DEV=true" >> $(venv_name)/bin/activate; \
	fi
	source $(venv_name)/bin/activate;

configure:
	$(local_python) -m pip install virtualenv

install: venv
	$(python) -m pip install -r $(deps);

run: venv
	time $(python) -W ignore $(run_script)

tests_path = ./tests/
pytest:
	$(python) -m pytest ${tests_path}

wily_reports:
	wily report src

FORCE:

clean:
	rm -rf $(venv_name)


# ##############################
# Python - Docker
# ##############################

image = "ml-model"
image_tag = $(shell docker images --format "{{.Tag}}" $(image) | head -n 1)
image_tag = "latest"

dockerfile = Dockerfile

docker-build:
	docker build \
		-t ${image}:${image_tag} \
		-f ${dockerfile} \
		.

operation = "learn"

docker-run:
	time docker run -it --rm \
		--network "host" \
		-v ${PWD}:/app \
		-e LOCAL_DEV='true' \
		${image}:${image_tag} \
		python -W ignore $(run_script)


# ##############################
# Jupyter Notebook
# ##############################
notebook:
	jupyter lab notebooks/ > /tmp/jupyterlab.log 2>&1 &

notebook_save:
	jupytext --to py notebooks/prediction_models_analysis.ipynb
	jupytext --to py notebooks/dataset_analysis.ipynb

notebook_load:
	jupytext --to notebook notebooks/prediction_models_analysis.py
	jupytext --to notebook notebooks/dataset_analysis.py


# ##############################
# AWS
# ##############################
.PHONY: local-ecr-login local-ecr-push pum

aws_profile = "<< profile name >>"
export AWS_PROFILE := $(aws_profile)
aws_account_id = $(shell export AWS_PROFILE; aws sts get-caller-identity | jq -r '.Account')
aws_region = eu-west-1

pum_script = ~/bin/pum-aws.py

pum:
	$(local_python) $(pum_script) -p $(aws_profile)

local-ecr-login:
	aws ecr get-login-password \
		--region $(aws_region) \
		| docker login \
		--username AWS --password-stdin \
		$(aws_account_id).dkr.ecr.$(aws_region).amazonaws.com

local-ecr-push:
	docker tag $(image):$(image_tag) $(aws_account_id).dkr.ecr.$(aws_region).amazonaws.com/$(image):$(image_tag)
	docker push $(aws_account_id).dkr.ecr.$(aws_region).amazonaws.com/$(image):$(image_tag)
	docker rmi $(aws_account_id).dkr.ecr.$(aws_region).amazonaws.com/$(image):$(image_tag)

# Lambda
#
.PHONY: lambda_install lambda_build lambda_run lambda_push lambda_clean

name = "<< name >>"
S3_BUCKET =
S3_KEY = 

lambda_install:
	pip3 install -r requirements.txt --system --target ./package
	cd package && zip -r9 ../$(name).zip *

lambda_build:
	zip -g $(name).zip src/

lambda_run:
	aws lambda invoke --function-name $(name) out --log-type Tail --query 'LogResult' --output text |  base64 -d

lambda_push: build
	aws lambda update-function-code --function-name $(name) --zip-file fileb://$(name).zip

lambda_push-s3:
	aws s3 cp $(name).zip s3://$(S3_BUCKET)
	aws lambda update-function-code --function-name $(name) --s3-bucket $(S3_BUCKET) --s3-key $(S3_KEY) --publish

lambda_clean:
	rm -rf package
