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
