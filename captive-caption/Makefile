create_environment:
	@echo "Creating environment"
	python -m venv venv

install_requirements:
	@echo "Installing requirements"
	venv/bin/pip install -r ./requirements.txt

configure_jupyter:
	# ipywidgets
	venv/bin/jupyter nbextension enable --py widgetsnbextension

	# nbautoexport
	venv/bin/nbautoexport install
	mkdir -p notebooks
	venv/bin/nbautoexport configure notebooks