// RUN: %{ispc} %s --target=sse4.1-i32x4 --arch=x86-64 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: not %{ispc} %s --target=sse4.1-i32x4 --arch=x86-64 --nostdlib --nowrap -DDEPENDENT_TYPE -o %t.o 2>&1 | FileCheck %s --check-prefix=ERROR --implicit-check-not=FATAL
// RUN: %{ispc} %s --target=sse4.1-i32x4 --arch=x86-64 --nostdlib --nowrap -DDEBUG_DUMP --debug -o %t.o > %t.log 2>&1
// RUN: FileCheck %s --check-prefix=DEBUG --implicit-check-not=Error: < %t.log
// REQUIRES: X86_ENABLED

#ifdef DEPENDENT_TYPE
template <typename T>
void unknown_intrinsic_type(T arg) {
    // ERROR: Error: Array size must be a compile-time constant or template non-type parameter.
    uniform int arr[sizeof(@llvm.ctpop(arg))];
}
#elif defined(DEBUG_DUMP)
template <typename T>
uniform int intrinsic_call(T arg) {
    // DEBUG-DAG: [unresolved] llvm.ctpop
    return @llvm.ctpop(arg);
}

template <int N>
uniform int leaf() { return N; }

uniform int consume(uniform int n) { return n; }
uniform float consume(uniform float n) { return n; }

uniform int local_template_argument() {
    constexpr uniform int n = 3;
    // DEBUG-DAG: [unresolved] leaf
    // DEBUG-DAG: [unresolved] consume
    return consume(1 + (2 + leaf<n>()));
}

uniform int test_intrinsic() { return intrinsic_call<uniform int>(3); }
#else
template <int N>
uniform int intrinsic_array_size() {
    uniform int arr[sizeof(@llvm.x86.sse41.ptestz((uniform int64<2>)N, (uniform int64<2>)1))];
    uniform int rounded[sizeof(@llvm.x86.sse41.round.ps((varying float)1.0f, N))];
    return sizeof(arr) + sizeof(rounded);
}

// CHECK-LABEL: @test
// CHECK: ret i32 80
uniform int test() { return intrinsic_array_size<3>(); }
#endif
