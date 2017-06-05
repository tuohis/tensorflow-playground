SHELL := /bin/bash

VIRTUALENV := .venv3
VENV_ACTIVATE := source ${VIRTUALENV}/bin/activate

${VIRTUALENV}:
	virtualenv -p python3 --system-site-packages ${VIRTUALENV}
	${VENV_ACTIVATE} && pip install tensorflow

curve_fitter: ${VIRTUALENV}
	${VENV_ACTIVATE} && python main.py

clean:
	rm -rf ${VIRTUALENV}
