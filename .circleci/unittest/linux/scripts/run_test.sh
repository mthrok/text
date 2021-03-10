#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env

# Do not run pytest from the root directory of the repository.
# The checked out code will be added to the Python module search path and
# will shadow the development/instaled package.
cd test
pytest --cov=torchtext --junitxml=test-results/junit.xml -v --durations 20 torchtext_unittest
