CURRENT_DIR := $(shell pwd)

init: clean dependencies

dependencies:
	pipenv install --dev

clean:
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -f report.xml
	rm -f coverage.xml

flake8:
	pipenv run flake8

mypy:
	pipenv run mypy

pylint:
	pipenv run pylint interview_test api.py run_feature_to_csv.py run_train_model.py

lint: flake8 pylint mypy

test:
	pipenv run pytest

ci:
	pipenv run pytest --junitxml=report.xml --cov-report xml

docker:
	docker build --pull -f Dockerfile -t mysqldb .
	docker run -i -t -d  -v $(CURRENT_DIR)/MLE_Task:/opt/MLE_Task/ --name mysql_container  -p 3307:3306 mysqldb --secure-file-priv /opt/MLE_Task
