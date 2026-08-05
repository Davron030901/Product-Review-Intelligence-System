.PHONY: install data baseline transformer test api docker

install:
	pip install -r requirements.txt

data:
	python -m src.data.build_dataset

baseline:
	python -m src.training.train_baseline

transformer:
	python -m src.training.train_transformer --epochs 3

test:
	python -m pytest tests -q

api:
	uvicorn src.api.main:app --reload --port 8000

docker:
	docker compose up --build

all: data baseline test
