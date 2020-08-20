#!/usr/bin/env bash

set -e

python setup.py clean develop
cd benchmark
python create_cpp_pipeline.py
cd csrc
mkdir -p build
cd build
cmake .. -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_PREFIX_PATH=/scratch/moto/libtorch/libtorch
cmake --build . --config RELEASE
./benchmark-jit_pipeline ../../jit_pipeline.pt ../../test.csv
