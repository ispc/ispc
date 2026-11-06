// Verifies casts between integers and pointers to SOA data: a pointer to a
// SOA struct or short vector can't be cast to an integer, and an integer can
// be cast to a SOA pointer with a zero slice offset. Null and zero constants
// initialize uniform and varying SOA pointers, but the address of SOA data is
// not a constant initializer.

// RUN: not %{ispc} %s --target=avx2-i32x8 --nowrap -O0 -o /dev/null -DERRORS 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
// RUN: not %{ispc} %s --target=avx2-i32x8 --nowrap -O2 -o /dev/null -DERRORS 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap -O0 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap -O2 -o /dev/null
// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86 --nowrap -O0 --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK-32

// REQUIRES: X86_ENABLED

struct P3 { float x, y, z; };
struct V3 { float<3> v; };
struct A4 { float a[4]; };

// CHECK: @gnull = global { <8 x i64>, <8 x i32> } zeroinitializer
// CHECK: @gzero = global { <8 x i64>, <8 x i32> } zeroinitializer
// CHECK: @gcast = global { <8 x i64>, <8 x i32> } zeroinitializer
// CHECK: @gunif = global { ptr, i32 } zeroinitializer
// CHECK: @garr = global [2 x { <8 x i64>, <8 x i32> }] zeroinitializer
// CHECK: @ghold = global %{{.*}} { { <8 x i64>, <8 x i32> } zeroinitializer, <8 x i32> {{.*}} }
// CHECK-32: @gnull = global { <8 x i32>, <8 x i32> } zeroinitializer
soa<8> P3 * varying gnull = NULL;
soa<8> P3 * varying gzero = 0;
soa<8> P3 * varying gcast = (soa<8> P3 * varying)0;
soa<8> P3 * uniform gunif = NULL;
soa<8> P3 * varying garr[2] = {NULL, 0};
struct Hold { soa<8> P3 * varying p; int k; };
Hold ghold = {NULL, 3};

#ifdef ERRORS
soa<8> P3 gs[16];
// CHECK-ERR: Initializer for global variable "gaddr" must be a constant
soa<8> P3 * varying gaddr = &gs[1];
// CHECK-ERR: Initializer for global variable "gaddru" must be a constant
soa<8> P3 * uniform gaddru = &gs[1];

// CHECK-ERR: Can't cast from pointer to SOA type "soa<8> float[4] * slice uniform" to integer type "uniform int64"
uniform int64 array_off(soa<8> A4 * uniform p) { return (uniform int64)(&p[1].a); }

// CHECK-ERR: Can't cast from pointer to SOA type "soa<8> struct P3 * slice uniform" to integer type "uniform int64"
uniform int64 struct_off(soa<8> P3 * uniform p) { return (uniform int64)(&p[1]); }

// CHECK-ERR: Can't cast from pointer to SOA type "soa<8> float<3> * slice uniform" to integer type "uniform int64"
uniform int64 vector_off(soa<8> V3 * uniform p) { return (uniform int64)(&p[1].v); }

// CHECK-ERR: Can't cast from pointer to SOA type "soa<8> struct P3 * slice varying" to integer type "varying int64"
varying int64 varying_off(soa<8> P3 * uniform p) { return (varying int64)(&p[programIndex]); }
#endif

// Pointers to SOA members of basic types can still be cast to integers.
// CHECK-LABEL: define i64 @"member_off___
// CHECK: ptrtoint ptr
uniform int64 member_off(soa<8> P3 * uniform p) { return (uniform int64)(&p[1].y); }

// CHECK-LABEL: define i64 @"elem_off___
// CHECK: ptrtoint ptr
uniform int64 elem_off(soa<8> A4 * uniform p) { return (uniform int64)(&p[1].a[2]); }

void g(soa<8> P3 * uniform a) {}
void gv(soa<8> P3 * varying a) {}

// CHECK-LABEL: define void @h(
// CHECK: inttoptr i64 %{{.*}} to ptr
// CHECK: insertvalue { ptr, i32 } poison, ptr %{{.*}}, 0
// CHECK: insertvalue { ptr, i32 } %{{.*}}, i32 0, 1
export void h(uniform int64 v) { g((soa<8> P3 * uniform)v); }

// CHECK-LABEL: define void @hv(
// CHECK: insertvalue { <8 x i64>, <8 x i32> } poison, <8 x i64> %{{.*}}, 0
// CHECK: insertvalue { <8 x i64>, <8 x i32> } %{{.*}}, <8 x i32> zeroinitializer, 1
export void hv(uniform int64 v) { gv((soa<8> P3 * varying)(v + programIndex)); }

// CHECK-LABEL: define void @h32(
// CHECK: inttoptr i32 %{{.*}} to ptr
export void h32(uniform int32 v) { g((soa<8> P3 * uniform)v); }

// A null pointer passed as a varying SOA pointer is smeared as a slice pointer.
// CHECK-LABEL: define void @hnull(
// CHECK: insertvalue { <8 x i64>, <8 x i32> }
export void hnull() {
    gv(0);
    gv(NULL);
}

// CHECK-LABEL: define void @hinit(
// CHECK: store { <8 x i64>, <8 x i32> } zeroinitializer
export void hinit() {
    soa<8> P3 *varying q = (soa<8> P3 * uniform)0;
    gv(q);
}

// CHECK-LABEL: define void @hstatic(
// CHECK: load { <8 x i64>, <8 x i32> }, ptr @static.{{.*}}.s
export void hstatic() {
    static soa<8> P3 *varying s = NULL;
    gv(s);
}

// A non-SOA uniform pointer cast to a varying SOA pointer gets a zero offset.
// CHECK-LABEL: define void @hptr(
// CHECK: insertvalue { ptr, i32 } %{{.*}}, i32 0, 1
// CHECK: insertvalue { <8 x i64>, <8 x i32> } poison, <8 x i64> %{{.*}}, 0
// CHECK: insertvalue { <8 x i64>, <8 x i32> } %{{.*}}, <8 x i32> %{{.*}}, 1
export void hptr(uniform float *uniform p) { gv((soa<8> P3 * varying)p); }
