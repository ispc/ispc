// Test for correct boolean return value representation in exported functions
// Regression test for issue where uniform bool return values could be 255 instead of 1

// RUN: %{ispc} -O2 --pic --target=avx2-i32x8 -h %t.h %s -o %t.o
// RUN: %{cc} -O2 -x c++ -c %s -o %t.cpp.o --include %t.h
// RUN: %{cc} %t.o %t.cpp.o -o %t.cpp.bin -lstdc++
// RUN: %t.cpp.bin | FileCheck %s

// CHECK: Raw ISPC result: 1 (as int)
// CHECK: Raw ISPC result: 1 (as bool)
// CHECK: asCppBool result: 1 (as int)
// CHECK: asCppBool result: 1 (as bool)

// REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_19_0+

#ifdef ISPC
export uniform bool foo(uniform float scl, uniform uint32 &isectsEqual) {
    uniform bool success = true;
    uint32 localIsectsEqual = 0;

    for (uniform uint32 y = 0; y < 2; y++) {
        float testValue = (float)y * scl;

        bool passed = true;
        bool asserted = false;

        if (!asserted) {
            if (testValue > 1.0f) {
                passed = false;
                if (testValue > 0.1f) {
                    asserted = true;
                }
            }
        }

        if (asserted) {
            success = false;
        }

        localIsectsEqual += passed ? 1 : 0;
    }

    isectsEqual = reduce_add(localIsectsEqual);

    return success;
}
#else
#include <iostream>
using namespace ispc;

inline bool asCppBool(bool ispcBool) {
    int i = ispcBool ? 1 : 0;
    return (bool)i;
}

int main() {
    float scl = 0.001f;
    uint32_t isectsEqual = 0;

    bool rawResult = foo(scl, isectsEqual);

    bool convertedResult = asCppBool(rawResult);

    std::cout << "Raw ISPC result: " << (int)rawResult << " (as int)" << std::endl;
    std::cout << "Raw ISPC result: " << rawResult << " (as bool)" << std::endl;
    std::cout << "asCppBool result: " << (int)convertedResult << " (as int)" << std::endl;
    std::cout << "asCppBool result: " << convertedResult << " (as bool)" << std::endl;

    return 0;
}
#endif
