#!/bin/bash -e
cd build
bin/check_isa
bin/ispc --support-matrix
if [[ $OSTYPE == 'darwin'* ]]; then
  cmake --build . --target check-all
else
  cmake --build . --target check-all ispc_benchmarks test
fi
