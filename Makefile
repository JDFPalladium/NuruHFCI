install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt


lint: 
	pylint --disable=R,C app.py

test:
	PYTHONPATH=. pytest -vv tests/test_app.py