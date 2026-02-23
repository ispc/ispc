// This test checks that generated code does not try to divide by zero.

// RUN: %{ispc} --pic --target=host -h %t.h %s -o %t.o
// RUN: %{cc} -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.c.bin
// RUN: %t.c.bin | FileCheck %s

// REQUIRES: !MACOS_HOST
// ISPC assertion failure in lAllDivBaseEqual (llvmutil.cpp) for generic
// targets: the integer divide optimization passes a non-power-of-2 baseValue.
// TODO: Fix this ISPC bug for generic targets.
// XFAIL: PPC64LE_HOST

// CHECK-NOT: Floating point exception
// CHECK: Correct result

#ifdef ISPC
struct Array {
    void* data_ptr_;
    int64 byte_size_;
};

struct State {
    Array input0_;
    Array input1_;
    Array output_;
};

static inline uint8* GetDataPtr(const uniform Array* const uniform array,
                                int64 workload_idx) {
    return (uint8*)(array->data_ptr_) + workload_idx * array->byte_size_;
}

#define TEMPLATE(T)                                                     \
    static inline void DoDiv(int64 idx, uniform State* uniform state) { \
        const T* lhs = (const T*)GetDataPtr(&(state->input0_), idx);    \
        const T* rhs = (const T*)GetDataPtr(&(state->input1_), idx);    \
        T* dst = (T*)GetDataPtr(&(state->output_), idx);                \
        *dst = *lhs / *rhs;                                             \
    }

#pragma ignore warning(perf)
TEMPLATE(int32)

#undef TEMPLATE

export void DoDivKernel(uniform int64 N, uniform State* uniform state) {
    foreach (i = 0...N) { DoDiv(i, state); }
}
#else
#include <stdio.h>

int main() {
    int a[6] = {4, 6, 8, 10, 12, 14};
    int b[6] = {2, 3, 4, 5, 6, 7};

    struct State state;

    state.input0_.data_ptr_ = a;
    state.input0_.byte_size_ = 4;

    state.input1_.data_ptr_ = b;
    state.input1_.byte_size_ = 4;

    state.output_.data_ptr_ = a;
    state.output_.byte_size_ = 4;

    DoDivKernel(6, &state);

    if (!(a[0] == 2 && a[1] == 2 && a[2] == 2 && a[3] == 2 && a[4] == 2 && a[5] == 2)) {
        return 1;
    }

    printf("Correct result\n");

    return 0;
}
#endif // ISPC
