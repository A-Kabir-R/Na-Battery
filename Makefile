# One-command entry points for the Na-Battery study.
#
# `make paper` is the reproduction target: it refuses to run on a dirty tree,
# rebuilds every artifact in dependency order, and stops at the first failure.
# A pipeline that silently continues past a broken stage produces a results
# directory that looks complete and is not.
#
# Override the interpreter or data locations from the command line, e.g.
#     make features PYTHON=python3 SIB_ROOT=/mnt/data/standard_cycling

PYTHON      ?= .venv/bin/python
ARTIFACTS   ?= $(CURDIR)/artifacts
RESULTS     := $(ARTIFACTS)/results
# Every PINN target trains on CUDA. Override only to debug on a machine with no
# GPU -- a CPU PINN run is days long and must never produce a reported number.
SIB_DEVICE  ?= cuda
PRIMARY_SEED ?= 42
export SIB_ARTIFACTS := $(ARTIFACTS)
# Without this the classical runner writes its log to paths.logs, which defaults
# to the external dataset drive -- outside the tree `clean-results` manages and
# outside the bundle the pod uploads.
export SIB_LOGS := $(ARTIFACTS)/logs

.DEFAULT_GOAL := help
.PHONY: help venv lock test test-fast lint canonical features classical pinn \
        pinn-nested pinn-ablation generalization low-data uncertainty \
        robustness studies aggregate figures clean-results check-clean paper

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtualenv and install runtime + test dependencies
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e ".[test]"

lock:  ## Freeze the resolved environment (run inside the venv you ship)
	$(PYTHON) -m pip freeze --all > requirements.lock.txt
	@echo "wrote requirements.lock.txt -- commit it with the results it produced"

test:  ## Full test suite
	$(PYTHON) -m pytest tests/ -q

test-fast:  ## Skip the slow PINN training tests
	$(PYTHON) -m pytest tests/ -q -k "not synthetic and not trainer"

lint:  ## Ruff
	$(PYTHON) -m ruff check src scripts tests

canonical:  ## Rebuild canonical tables from raw .ird (needs the dataset mounted)
	$(PYTHON) scripts/00_build_canonical.py --write-samples

features:  ## Build the unified anchor table
	$(PYTHON) scripts/01_build_features.py

classical:  ## Classical models over the locked grouped folds
	$(PYTHON) scripts/02_run_all_experiments.py --tuned
	$(PYTHON) scripts/03_aggregate_results.py

pinn:  ## NaPINN-Q across folds and seeds
	SIB_DEVICE=$(SIB_DEVICE) $(PYTHON) scripts/06_run_pinn_experiments.py
	$(PYTHON) scripts/07_aggregate_pinn_results.py

pinn-nested:  ## Select the refit epoch count on all inner folds, then refit
	SIB_DEVICE=$(SIB_DEVICE) $(PYTHON) scripts/15_run_nested_pinn_selection.py
	$(PYTHON) scripts/07_aggregate_pinn_results.py

pinn-ablation:  ## Physics ablations (trains PINNs; GPU)
	SIB_DEVICE=$(SIB_DEVICE) $(PYTHON) scripts/10_run_pinn_ablations.py \
	  --seed $(PRIMARY_SEED) --preprocessing unified

generalization:  ## LOCO + factor-level-out study
	$(PYTHON) scripts/11_run_generalization_study.py

low-data:  ## Learning curves in units of training cells
	$(PYTHON) scripts/12_run_low_data_study.py

uncertainty:  ## Cell-grouped conformal intervals + coverage by condition
	$(PYTHON) scripts/13_run_uncertainty_study.py

robustness:  ## Input perturbations; checks that intervals widen
	$(PYTHON) scripts/14_run_robustness_study.py

studies: generalization low-data uncertainty robustness  ## All four study scripts

aggregate:  ## Combine classical and PINN arms onto shared folds
	$(PYTHON) scripts/09_combine_classical_pinn_results.py

figures:  ## Regenerate every figure from the frozen prediction tables
	$(PYTHON) scripts/04_plot_results.py
	$(PYTHON) scripts/08_plot_pinn_results.py
	$(PYTHON) scripts/16_make_paper_figures.py

clean-results:  ## Delete derived results (keeps canonical and features)
	rm -rf $(RESULTS)

check-clean:  ## Fail if the working tree is dirty
	@if [ -n "$$(git status --porcelain)" ]; then \
	  echo "ERROR: working tree is dirty. A result you cannot tie to a commit"; \
	  echo "       is a result you cannot defend. Commit or stash first:"; \
	  git status --porcelain; \
	  exit 1; \
	fi
	@echo "clean tree at $$(git rev-parse HEAD)"

# Prerequisites run left to right, so this target is order-dependent and must
# not be built in parallel: `make -j paper` would start `classical` before
# `features` exists.
.NOTPARALLEL:

# Execution order: pinn first (five-seed sweep), then pinn-nested (epoch
# selection + seed-42 refit), then pinn-ablation, then aggregate.
# pinn-nested must run AFTER pinn so that nested selection replaces only the
# seed-42 checkpoint and the remaining seeds are already present for
# seed-sensitivity reporting.
paper: check-clean test features classical pinn pinn-nested pinn-ablation \
       studies aggregate figures  ## Full reproduction
	@echo
	@echo "reproduction complete at commit $$(git rev-parse HEAD)"
	@echo "results: $(RESULTS)"
