#!/bin/bash

run_clang_format() {
    clang-format-10 -style=file -i $1
}

for f in include/ispc/*.h; do
    run_clang_format $f
done

for f in src/*.h; do
    run_clang_format $f
done

for f in src/*.cpp; do
    run_clang_format $f
done
