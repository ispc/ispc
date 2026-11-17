// Ensure that output(-dM) can output the embedded predefined ISPC macros with empty input
// RUN: %{ispc} -dM --outfile=emptydm.dM.cpp -
// RUN: FileCheck %s --input-file=emptydm.dM.cpp --check-prefix=EMPTYDM
// EMPTYDM: ISPC

// Ensure that the output(-dM) includes macros but doesn't include functions
// -dm itself doesn't enable "ShowComments" so only need to check for 1 count.
// RUN: %{ispc} %s -dM --outfile=%t.dM.cpp
// RUN: FileCheck %s --input-file=%t.dM.cpp --check-prefix=DM
// DM: MyDescriptor
// DM: MyType
// DM: MyValue
// DM-NOT: my_func

// Intrinsic tokens must not end macro scanning, even without the implicit stdlib.
// RUN: %{ispc} %s --target=host --nostdlib --nowrap -DMACROS_AFTER_INTRINSIC -dM -o - | FileCheck %s --check-prefix=INTRINSIC
// INTRINSIC-DAG: #define AfterIntrinsic 1
// INTRINSIC-DAG: #define DumpEmpty {{ *$}}
// INTRINSIC-DAG: #define DumpFunction(a,b) a ## b
// INTRINSIC-DAG: #define DumpGNU(args...) args
// INTRINSIC-DAG: #define DumpOdd(__VA_ARGS__,last) last
// INTRINSIC-DAG: #define DumpString(a) #a
// INTRINSIC-DAG: #define DumpVariadic(first,...) first,__VA_ARGS__
// INTRINSIC-DAG: #define ObjCFeature 0

#ifdef MACROS_AFTER_INTRINSIC
uniform float intrinsic_value(uniform int bits) {
    return @llvm.ispc.bitcast(bits, (uniform float)0);
}
#define AfterIntrinsic 1
#define DumpEmpty
#define DumpFunction(a,b) a ## b
#define DumpGNU(args...) args
#define DumpOdd(__VA_ARGS__,last) last
#define DumpString(a) #a
#define DumpVariadic(first,...) first,__VA_ARGS__
#if __has_feature(objc_fixed_enum)
#define ObjCFeature 1
#else
#define ObjCFeature 0
#endif
#endif

// Ensure that output(-E) has functions but doesn't include macros
// -E has "ShowComments", so need to count lit comments
// RUN: %{ispc} %s -E --outfile=%t.E.cpp
// RUN: FileCheck %s --input-file=%t.E.cpp --check-prefix=E
// E: CHECK-COUNT-3: MyDescriptor
// E: CHECK-COUNT-3: MyType
// E: CHECK-COUNT-3: MyValue
// E: CHECK-COUNT-4: my_func

// Ensure that output(-dD) has both functions and macros.
// -dD has "ShowComments", so need to count lit comments
// RUN: %{ispc} %s -dD --outfile=%t.dD.cpp
// RUN: FileCheck %s --input-file=%t.dD.cpp --check-prefix=DD
// DD: CHECK-COUNT-4: MyDescriptor
// DD: CHECK-COUNT-4: MyType
// DD: CHECK-COUNT-4: MyValue
// DD: CHECK-COUNT-4: my_func

#define MyType float
#define MyDescriptor const
#define MyValue 100

MyType my_func(MyDescriptor MyType _value) { return _value + MyValue; }
