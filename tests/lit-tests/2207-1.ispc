// This test checks that the compiler correctly parses integer suffixes.

// RUN: %{ispc} --target=host --emit-llvm-text --nostdlib --nowrap -g %s -o - 2>&1 | FileCheck %s

// CHECK-NOT: Error:
// CHECK-NOT: FATAL ERROR:

uniform int test()
{
    uniform int arr[] = {
        15,
        15u,
        15l,
        15ul,
        15uL,
        15Ul,
        15UL,
        15ull,
        15Ull,
        15ULL,
        15uLL,
        15llu,
        15llU,
        15LLU,
        15LLu,
        15G,
        15k,
        15G,
        15Gu,
        15Gl,
        15Mull,
        15MUll,
        15MULL,
        15MuLL,
        15Mllu,
        15MllU,
        15MLLU,
        15MLLu,
        15kull,
        15kUll,
        15kULL,
        15kuLL,
        15kllu,
        15kllU,
        15kLLU,
        15kLLu,
        15Gull,
        15GUll,
        15GULL,
        15GuLL,
        15Gllu,
        15GllU,
        15GLLU,
        15GLLu
    };
    return arr[0];
}

