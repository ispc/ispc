// Make sure the appropriate ssp function attribute is set.

// RUN: %{ispc} %s --emit-llvm-text -o - | FileCheck %s -check-prefixes=CHECK_ALL,DEFAULT
// RUN: %{ispc} %s --emit-llvm-text --stack-protector -o - | FileCheck %s -check-prefixes=CHECK_ALL,SSPON
// RUN: %{ispc} %s --emit-llvm-text --stack-protector=on -o - | FileCheck %s -check-prefixes=CHECK_ALL,SSPON
// RUN: %{ispc} %s --emit-llvm-text --stack-protector=none -o - | FileCheck %s -check-prefixes=CHECK_ALL,DEFAULT
// RUN: %{ispc} %s --emit-llvm-text --stack-protector=strong -o - | FileCheck %s -check-prefixes=CHECK_ALL,SSPSTRONG
// RUN: %{ispc} %s --emit-llvm-text --stack-protector=all -o - | FileCheck %s -check-prefixes=CHECK_ALL,SSPALL

// CHECK_ALL:   define {{.*}}foo{{.*}}#[[ATTRNO:[0-9]+]]
// DEFAULT-NOT: attributes #[[ATTRNO]] = { {{.* ssp}} }
// SSPON:       attributes #[[ATTRNO]] = { {{.* ssp .*}} }
// SSPSTRONG:   attributes #[[ATTRNO]] = { {{.* sspstrong .*}} }
// SSPALL:      attributes #[[ATTRNO]] = { {{.* sspreq .*}} }

int foo( int val ) {
    return val;
}
