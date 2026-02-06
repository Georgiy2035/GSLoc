# ============================================================================ #
# _____________________________ COMMON VARIABLES _____________________________ #
# ============================================================================ #

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))
ROOT_DIR := $(MKFILE_DIR)

SHELL := /usr/bin/bash

docker_compose_files := \
	-f docker/docker-compose.yaml

# ============================================================================ #
# _________________________________ COMMANDS _________________________________ #
# ============================================================================ #
docker_build := source $(ROOT_DIR)/.envrc && docker compose $(docker_compose_files) build
docker_down := source $(ROOT_DIR)/.envrc && docker compose $(docker_compose_files) down
docker_run_interactive := source $(ROOT_DIR)/.envrc && docker compose $(docker_compose_files) run -it --rm
docker_up := source $(ROOT_DIR)/.envrc && docker compose $(docker_compose_files) up

# ============================================================================ #
# _______________________________ BASIC RECIPES ______________________________ #
# ============================================================================ #

# ---------------------------------------------------------------------------- #
#  Specific build rules
# ---------------------------------------------------------------------------- #
build-mmpr:
	@echo "Building"
	@cd $(ROOT_DIR) && $(docker_build) multi-modal-place-recognition

# ============================================================================ #
# ________________________________ RUN RECIPES _______________________________ #
# ============================================================================ #
run-interactive-mmpr:
	@echo "Runnning and attaching to $(SERVICE)"
	@cd $(ROOT_DIR) \
	&& $(docker_run_interactive) multi-modal-place-recognition $(CMD)

run-juputernotebook-mmpr:
	@echo "Runnning and attaching to $(SERVICE)"
	@cd $(ROOT_DIR) \
	&& $(docker_run_interactive) multi-modal-place-recognition-notebooks 
# ============================================================================ #
# _____________________________ AUXILLARY RECIPES ____________________________ #
# ============================================================================ #

prepare-terminal-for-visualization:
	DISPLAY=$(DISPLAY) xhost +local:
	DISPLAY=$(DISPLAY) xhost +
	RCUTILS_COLORIZED_OUTPUT=1

prepare-git-repo:
	@echo "Install pre-commit"
	pip3 install pre-commit
	pre-commit install
