APP_NAME := synth
VENV := .venv
BIN := $(VENV)/bin
PY_VERSION := python3

.DEFAULT_GOAL := help

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: venv
venv: 
	virtualenv venv -p python3

$(BIN)/activate:
	$(PY_VERSION) -m venv --prompt $(APP_NAME) $(VENV)

.PHONY: install
install: venv upgrade-pip ## Install Python dependencies
	./venv/bin/python -m pip install -r requirements.txt

.PHONY: upgrade-pip
upgrade-pip: venv ## Upgrade pip and related
	./venv/bin/python -m pip install --upgrade pip wheel setuptools pip-tools
