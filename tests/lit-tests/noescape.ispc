// RUN: %{ispc} --target=host --nowrap --nostdlib --debug-phase=2:2 %s --no-discard-value-names -o t.o | FileCheck %s

// CHECK: declare void @foo___un_3C_uni_3E_({{.*}} {{nocapture|captures\(none\)}}, {{.*}}
void foo(__attribute__((noescape)) uniform int *uniform ptr);

// CHECK: declare {{.*}} @boo___un_3C_uni_3E_({{.*}} {{nocapture|captures\(none\)}}, {{.*}}
uniform int8 *uniform boo(__attribute__((noescape)) uniform int32 *uniform x);

// CHECK: declare void @arr___un_3C_uni_3E_({{.*}} {{nocapture|captures\(none\)}}, {{.*}}
void arr(__attribute__((noescape)) uniform int x[]);

// CHECK: declare void @struct____{{.*}}({{.*}} {{nocapture|captures\(none\)}}, {{.*}}
typedef struct {
    int x;
} my_struct;
void struct_(__attribute__((noescape)) uniform my_struct *uniform y);
