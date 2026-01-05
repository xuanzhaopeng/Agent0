#!/bin/bash

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3
/workspace/miniconda3/bin/conda init bash && source ~/.bashrc
rm Miniconda3-latest-Linux-x86_64.sh

# Install poetry
export POETRY_HOME=/workspace/.poetry
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="/workspace/.poetry/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
poetry config virtualenvs.in-project true



