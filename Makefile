install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
		
format:
	black *.py
	
lint:
	pylint --disable=R,C,W0212,W0105 *.py
	
all: install lint format
