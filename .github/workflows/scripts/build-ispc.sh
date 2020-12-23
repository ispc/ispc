#!/bin/bash -e
echo PATH=$PATH
mkdir build && cd build
cmake .. -DISPC_PREPARE_PACKAGE=ON -DISPC_INCLUDE_BENCHMARKS=ON
make -j4 package
