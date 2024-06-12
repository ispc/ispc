// RUN: echo "" | %{ispc} -o - --target=avx512skx-x16 -E --emit-asm -

// REQUIRES: LINUX_HOST && X86_ENABLED
