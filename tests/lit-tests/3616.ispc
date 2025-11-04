// Verifies that casting a SOA element address to integer doesn't crash the compiler.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s

struct Point { uint64 x; float y; float z; };

// Test uniform slice pointer to uniform integer
// CHECK-LABEL: define i64 @"gp___
// CHECK: extractvalue { ptr, i32 }
// CHECK: extractvalue { ptr, i32 }
// CHECK: getelementptr
// CHECK: getelementptr
// CHECK: ptrtoint ptr
// CHECK: ret i64

uniform uint64 gp(soa<16> Point v[]) {
    return (uniform uint64)&v[0].x;
}

// Test uniform slice pointer to varying integer
// CHECK-LABEL: define <{{[0-9]+}} x i64> @"vp___
// CHECK: extractvalue { ptr, i32 }
// CHECK: extractvalue { ptr, i32 }
// CHECK: getelementptr
// CHECK: getelementptr
// CHECK: ptrtoint ptr
// CHECK: insertelement <{{[0-9]+}} x i64> poison, i64
// CHECK: shufflevector <{{[0-9]+}} x i64>
// CHECK: ret <{{[0-9]+}} x i64>

varying uint64 vp(soa<16> Point v[]) {
    return (varying uint64)&v[0].x;
}

// Test uniform slice pointer to uniform 16-bit integer
// CHECK-LABEL: define i16 @"gp16___
// CHECK: extractvalue { ptr, i32 }
// CHECK: extractvalue { ptr, i32 }
// CHECK: getelementptr
// CHECK: getelementptr
// CHECK: ptrtoint ptr
// CHECK: trunc
// CHECK: ret i16

uniform uint16 gp16(soa<16> Point v[]) {
    return (uniform uint16)&v[0].x;
}