.PHONY: init clean env format lint precommit prod-requirements requirements symlink sync_raw_data test

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# can be 'cpu', 'gpu' or 'm1'
sys := "cpu"
name := "venv"

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

ifeq (,$(shell which mamba))
HAS_MAMBA=False
else
HAS_MAMBA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

# Initialize project (requirements + io/ folder)
init: requirements symlink precommit

# Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# Set up Python environment (requires `sys` variable to be set)
venv:
ifeq (True, $(HAS_MAMBA))
	@echo ">>> Detected mamba, creating conda environment via mamba."
	
	# Create the conda environment
	mamba env create --prefix=./$(name) -f requirements/env_$(sys).yml

	@echo ">>> New mamba env created. Activate from project directory with:\nconda activate ./$(name)"
else ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	
	# Create the conda environment
	conda env create --prefix=./$(name) -f requirements/env_$(sys).yml

	@echo ">>> New conda env created. Activate from project directory with:\nconda activate ./$(name)"
else
	@echo ">>> No conda detected. Please install conda or manually install requirements in your preferred python version."
endif

# Update dependencies
update_dep:
	cd ./requirements; \
	pip install -r prod.txt; \
	pip install -r dev.txt; \

# Format src directory using black
format:
	isort .
	autoflake -r --in-place --remove-unused-variables src
	black --config=pyproject.toml .

# Lint using pylint
lint:
ifdef VIRTUAL_ENV
	pylint src
else
	@echo "Please create your virtual environment and activate it first (make env; source env/bin/activate)."
endif

# Set up pre-commit hooks
configure_precommit:
	pip install pre-commit black pylint isort
	pre-commit install

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
