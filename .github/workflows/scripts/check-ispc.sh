#!/bin/bash -e
cd build
bin/check_isa
bin/ispc --support-matrix
cmake --build . --target check-all ispc_benchmarks test
