#!/bin/bash -e
cd build
bin/check_isa
bin/ispc --support-matrix
make check-all
make ispc_benchmarks && make test
