# ISPC micro-benchmarks.

Micro-benchmarks are designed to test efficiency of different aspects of ISPC code generation. This suit assumes three type of tests:

1. [**Individual language features and library functions**](01_trivial). Testing them in isolation focuses on implementation quality of language constructs, like ``foreach`` loops and library function like ``aos_to_soa()``. This category of tests allow covering ISPC functionality quite thoroughly, but doesn't address interplay between different features and compiler optimizations.
2. [**Feature combinations and non-obvious optimization effects**](02_medium). It's impractical to cover all combinations, so this kind of tests requires understanding of the specific effects triggered by features combination. An example might be memory operations nesting in non-trivial loops.
3. [**More complex cases inspired by real algorithms**](03_complex). Any example reduced from real-life applications and representing either a known performance problem, or a code snippet known to be critical for performance of the real code. When contributing this kind of examples, please mind the license of the original code.

The benchmarks might be used for comparison of different platforms and regression tracking. In this case ISPC implementation is sufficient.

For some features it also makes sense to compare with alternative implementations in other languages. In such case C++ (or other) implementation needs to be provided.

## Building and running

Use ``-DISPC_INCLUDE_BENCHMARKS=ON`` in your CMake configuration flags to enable building benchmarks. If ``--recurse-submodules`` was not used during repo cloning, CMake will fetch [Google Benchmark](https://github.com/google/benchmark) submodule. By default ISPC built from sources will used for building benchmarks. If you need to use pre-build ISPC binary, you can turn off building ISPC by setting ``-DISPC_BUILD=OFF`` and provide ISPC in your `PATH` environment variable.

You can use CMake options ``BENCHMARKS_ISPC_TARGETS`` and ``BENCHMARKS_ISPC_FLAGS`` to set specific target or ISPC compilation switches. For example, ``-DBENCHMARKS_ISPC_TARGETS=avx512skx-x8,avx2-i32x8 -DBENCHMARKS_ISPC_FLAGS="-O3 --woff"``. Note that using auto-dispatch (i.e. specifying more than one target on the command line) will noticeably affect short benchmark runs.

To run benchmarks, you need to execute them individually. They will be located in `benchmarks` folder of your install location. You can also run `make test` to verify that benchmarks are built correctly and executed successfully.

To do a performance measurements, it is recommended to do a frequency stabilization on your system. This is OS and hardware specific and there is no universal recipe, but here are the things that might work:
1. Disable CPU frequency scaling on Linux, as suggested in Google Benchmark [documentation](https://github.com/google/benchmark/blob/main/docs/user_guide.md#disabling-cpu-frequency-scaling)
2. Control the CPU core where the benchmark is running. This is especially important for heterogeneous CPUs. On Linux you can use `taskset` to pin the process to specific core. For example:
```
taskset --cpu-list 0 bin/01_aossoa
```

## TODO

### Individual language features and library functions.

- Loops: ``foreach`` loops vs ISPC ``for`` loops vs C++ ``for`` loops. Iteration limits divisible and not divisible by programIndex need to be tested.
- Nested loops: multi-dimensional ``foreach`` loops vs separate ``foreach`` loops vs ``for`` loops. Different axis for vectorization need to be tested.
- ``foreach_unique`` performance.
- ``foreach_active`` performance.
- memory operations: gather/scatter, vector load/store, masked load/store.
- ``aos_to_soa()``, ``soa_to_aos()`` functions.
- other ``stdlib`` function.
- ``select()`` vs ``cond ? t : f``.

### Feature combinations and non-obvious optimization effects.

- strided loads/stores, which combine to continuous loads/stores.
- dot-product and cross-product of short vectors.
- other code gen artifacts, triggered by seemingly similar code, but yielding different results.

