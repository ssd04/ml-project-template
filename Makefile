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

pum:
	$(local_python) $(pum_script) -p $(aws_profile)

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
