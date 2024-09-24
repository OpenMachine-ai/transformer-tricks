#!/bin/bash

# Make sure to increment the version number in pyproject.toml before
# running this script! See below link for more details:
# https://packaging.python.org/en/latest/tutorials/packaging-projects/
#
# Setup: install build and twine via 'pip3 install build twine'

# create folder 'pypi' and copy all relevant files
rm -rf pypi
mkdir pypi
mkdir -p pypi/transformer_tricks
cp LICENSE README.md pyproject.toml pypi
cp tricks.py pypi/transformer_tricks
touch        pypi/transformer_tricks/__init__.py

# build and upload
cd pypi
python3 -m build
python3 -m twine upload dist/*
