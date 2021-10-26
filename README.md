# ML Project

## Project structure

Makefile commands can be used to manage the project.

The application is containerized, so there is a `Dockerfile` for specifying how
the docker image will be constructed. The python requirements are in the
`requirements.txt` file.

Folder structure:
* `src`: application production code
* `tests`: unit tests folder; no tests at the moment
* `notebooks`: jupyter notebooks used for machine learning setup
* `conf`: config files for each ML model
* `scripts`: usefull scripts

## Data Source

## Models analysis

The jupyter notebooks can be used as a playground for the modelling part.

### Notebooks

Prerequisites:
* jupyterlab
* jupytext

 The notebooks are saved as python files, since it is better when keeping track
 with source control.

There are several commands for managing the notebooks from command line (the
commands can be checked directly in the makefile):
```bash
# open in browser
make notebooks

# save notebooks as python files
make notebook_save

# load notebooks from py files
make notebook_load
```

# Usefull links

* [Fastai course - Google Colab](https://course.fast.ai/start_colab)
* [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
* [Sklearn Documentation](https://scikit-learn.org/stable/getting_started.html)
