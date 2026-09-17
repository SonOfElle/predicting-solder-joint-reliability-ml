.PHONY: help env data notebooks figures metrics test clean all

help:
	@echo "Targets:"
	@echo "  env        Create or update the conda environment from environment.yml"
	@echo "  data       Regenerate the synthetic dataset into data/synthetic/"
	@echo "  notebooks  Execute all notebooks in notebooks/ in alphabetical order"
	@echo "  figures    Regenerate all figures into reports/figures/"
	@echo "  metrics    Regenerate results/metrics.csv"
	@echo "  test       Run the pytest suite"
	@echo "  clean      Remove generated artefacts and caches"
	@echo "  all        Run data, notebooks, figures, metrics, test"

env:
	conda env create -f environment.yml || conda env update -f environment.yml

data:
	python -m solder_reliability.gan

notebooks:
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

figures:
	python -m solder_reliability.viz

metrics:
	python -m solder_reliability.evaluate

test:
	pytest tests/

clean:
	rm -rf data/synthetic/*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf reports/figures/*
	rm -f results/metrics.csv

all: data notebooks figures metrics test