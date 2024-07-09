#!/bin/bash -e
#
# Copyright 2024, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cd build
bin/check_isa
bin/ispc --binary-type
bin/ispc --support-matrix
if [[ $OSTYPE == 'darwin'* ]]; then
  cmake --build . --target check-all
else
  ARCH=$(uname -m)
  if [ "$ARCH" == "aarch64" ]; then
    cmake --build . --target check-all
  else
    cmake --build . --target check-all ispc_benchmarks test
  fi
fi
