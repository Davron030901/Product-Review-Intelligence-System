.PHONY: install data eda baseline transformer transfer compare validate test api docker

install:
	pip install -r requirements.txt

data:
	python -m src.data.build_dataset

baseline:
	python -m src.training.train_baseline

transformer:
	python -m src.training.train_transformer --epochs 3

transfer:
	python -m src.training.evaluate_transfer

compare:
	python -m src.training.compare_models

eda:
	python notebooks/01_eda.py

validate-sample:
	python -m src.data.validate_labels sample --n 250

validate-score:
	python -m src.data.validate_labels score

test:
	python -m pytest tests -q

api:
	uvicorn src.api.main:app --reload --port 8000

docker:
	docker compose up --build

all: data eda baseline transfer compare test
