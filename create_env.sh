#!/bin/sh -xe
echo "#### CreateVirtualEnv.sh ####"

# Check for conda
if command -v conda &> /dev/null; then
    echo "### Creating conda environment ###"
    conda create -y -n transformerAction python=3.9
    conda activate transformerAction
else
    echo "### Creating pipenv environment ###"
    pip install --user pipenv
    pipenv --python 3.9
    pipenv shell
fi

echo "#### Installing requirements ####"
if [ "$CONDA_PREFIX" != "" ]; then
    conda install --file requirements.txt
else
    pip install -q -r requirements.txt
fi

echo "### Virtual environment setup complete ###"
echo "You are now in a Python virtual environment with the required packages installed."
