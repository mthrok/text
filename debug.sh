#!/usr/bin/env bash

set -e

rm -rf benchmark/csrc/build
mkdir -p benchmark/csrc/build

(
    ENABLE_PYBIND11=1 python setup.py clean develop
    cd benchmark
    python create_cpp_pipeline.py
)
(
    ENABLE_PYBIND11=0 python setup.py clean develop
    cd benchmark/csrc/build
    cmake .. -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_PREFIX_PATH=/scratch/moto/libtorch/libtorch
    cmake --build . --config RELEASE
    ./benchmark-jit_pipeline ../../jit_pipeline.pt ../../test.csv
)
