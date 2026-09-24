.PHONY: help env data notebooks figures metrics test clean all

PYTHON := python
NOTEBOOKS := \
	notebooks/01_eda.ipynb \
	notebooks/02_gan_synthesis.ipynb \
	notebooks/03_model_training_real.ipynb \
	notebooks/04_model_training_synthetic.ipynb \
	notebooks/05_results_comparison.ipynb

help:
	@echo "Targets:"
	@echo "  env        Create or update the conda environment from environment.yml"
	@echo "  data       Generate the synthetic dataset via 02_gan_synthesis.ipynb"
	@echo "  notebooks  Execute all notebooks in notebooks/ in order"
	@echo "  figures    Verify reports/figures/ contains the expected figures"
	@echo "  metrics    Verify results/ contains metrics.csv and cross_evaluation.csv"
	@echo "  test       Run the pytest suite"
	@echo "  clean      Remove generated artefacts and caches"
	@echo "  all        Run notebooks, figures, metrics, test"

env:
	conda env create -f environment.yml || conda env update -f environment.yml

data:
	jupyter nbconvert --to notebook --execute --inplace notebooks/02_gan_synthesis.ipynb

notebooks:
	jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOKS)

figures:
	@$(PYTHON) -c "from pathlib import Path; ps=sorted(Path('reports/figures').glob('*.png')); assert ps, 'no figures; run make notebooks first'; print(f'{len(ps)} figure(s) present in reports/figures/')"

metrics:
	@$(PYTHON) -c "from pathlib import Path; a=Path('results/metrics.csv'); b=Path('results/cross_evaluation.csv'); assert a.exists() and b.exists(), 'metrics missing; run make notebooks first'; print('metrics present: results/metrics.csv, results/cross_evaluation.csv')"

test:
	pytest tests/ -v

clean:
	rm -rf data/synthetic/fixed data/synthetic/original
	rm -f data/synthetic/*.npy
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache

all: notebooks figures metrics test
