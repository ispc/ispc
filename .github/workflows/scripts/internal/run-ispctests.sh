#!/bin/bash -e
git clone https://$ACCESS_TOKEN@github.com/intel-innersource/applications.gaming.ispctest
cd applications.gaming.ispctest/Scripts
./GenerateProjectFiles.sh

targets=(release avx2 avx512skx-i32x16 avx512skx-i32x8 sse2 sse4)
f_type="x86_64"
# Build targets
pushd ../Build/ISPCTest/gmake2/clang/
for target in ${targets}
do
    target="${target}_${f_type}"
    make config=$target clean
    make config=$target -j
done
popd

# Run tests
pushd ../x86_64/
for target in *
do
    ./$target --benchmark_min_time=0.1 --benchmark_repetitions=1 --benchmark_out=validate.json --benchmark_out_format=json --benchmark_report_aggregates_only=true
done
popd

