#!/bin/bash -e
echo PATH=$PATH
cmake -B build -DISPC_PREPARE_PACKAGE=ON -DISPC_INCLUDE_BENCHMARKS=ON
cmake --build build --target package -j4
