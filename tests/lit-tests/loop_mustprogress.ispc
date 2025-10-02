// Test to check that llvm.loop.mustprogress metadata is added to loops with non-constant conditions.

// RUN: %{ispc} %s --target=host --nowrap -O0 --emit-llvm-text --no-discard-value-names --nostdlib -o - | FileCheck %s

// CHECK-LABEL: define void @test_for_uniform___
// CHECK: br label %{{[a-zA-Z_][a-zA-Z0-9_]*}}, !llvm.loop [[LOOP_FOR_UNI:![0-9]+]]

// CHECK-LABEL: define void @test_for_varying___
// CHECK: br label %{{[a-zA-Z_][a-zA-Z0-9_]*}}, !llvm.loop [[LOOP_FOR_VARY:![0-9]+]]

// CHECK-LABEL: define void @test_infinite_for___
// CHECK-NOT: !llvm.loop

// CHECK-LABEL: define void @test_while_uniform___
// CHECK: br label %{{[a-zA-Z_][a-zA-Z0-9_]*}}, !llvm.loop [[LOOP_WHILE_UNI:![0-9]+]]

// CHECK-LABEL: define void @test_while_varying___
// CHECK: br label %{{[a-zA-Z_][a-zA-Z0-9_]*}}, !llvm.loop [[LOOP_WHILE_VARY:![0-9]+]]

// CHECK-LABEL: define void @test_infinite_while___
// CHECK-NOT: !llvm.loop

// CHECK-LABEL: define void @test_do_while_uniform___
// CHECK: br i1 %{{[a-zA-Z_][a-zA-Z0-9_]*}}, label %{{[a-zA-Z_][a-zA-Z0-9_]*}}, label %{{[a-zA-Z_][a-zA-Z0-9_]*}}, !llvm.loop [[LOOP_DO_UNI:![0-9]+]]

// CHECK-DAG: br i1 %{{[a-zA-Z_.0-9]+}}, label %do_loop, label %do_exit, !llvm.loop [[LOOP_DO_VARY:![0-9]+]]

// CHECK-DAG: br i1 %{{[a-zA-Z_.0-9]+}}, label %foreach_full_body, label %partial_inner_all_outer, !llvm.loop [[LOOP_FOREACH:![0-9]+]]

// Regular for loop - should have mustprogress
extern void sink(uniform float);
extern void sink(float);

void test_for_uniform(uniform int n) {
    for (uniform int i = 0; i < n; i++) {
        sink(i);
    }
}

void test_for_varying(uniform int n) {
    for (int i = 0; i < n; i+=programCount) {
        sink(i);
    }
}

// Infinite loop - should NOT have mustprogress (constant condition)
void test_infinite_for() {
    uniform int i = 0;
    for (;;) {
        sink(i);
        i++;
        if (i >= 10) break;
    }
}

// Regular while loop - should have mustprogress
void test_while_uniform(uniform int n) {
    uniform int i = 0;
    while (i < n) {
        sink(i);
        i++;
    }
}

void test_while_varying(uniform int n) {
    int i = 0;
    while (i < n) {
        sink(i);
        i+=programCount;
    }
}

// Infinite while loop - should NOT have mustprogress
void test_infinite_while(uniform int n) {
    while(true) {}
}

// Regular do-while - should have mustprogress
void test_do_while_uniform(uniform int n) {
    uniform int i = 0;
    do {
        sink(i);
        i++;
    } while (i < n);
}

void test_do_while_varying(uniform int n) {
    int i = 0;
    do {
        sink(i);
        i+=programCount;
    } while (i < n);
}

// Foreach loop - should have mustprogress
void test_foreach(uniform int n) {
    foreach (i = 0 ... n) {
        sink(i);
    }
}

// CHECK-DAG: [[LOOP_FOR_UNI]] = distinct !{[[LOOP_FOR_UNI]], [[MUSTPROGRESS:![0-9]+]]}
// CHECK-DAG: [[LOOP_FOR_VARY]] = distinct !{[[LOOP_FOR_VARY]], [[MUSTPROGRESS]]}
// CHECK-DAG: [[LOOP_WHILE_UNI]] = distinct !{[[LOOP_WHILE_UNI]], [[MUSTPROGRESS]]}
// CHECK-DAG: [[LOOP_WHILE_VARY]] = distinct !{[[LOOP_WHILE_VARY]], [[MUSTPROGRESS]]}
// CHECK-DAG: [[LOOP_DO_UNI]] = distinct !{[[LOOP_DO_UNI]], [[MUSTPROGRESS]]}
// CHECK-DAG: [[LOOP_DO_VARY]] = distinct !{[[LOOP_DO_VARY]], [[MUSTPROGRESS]]}
// CHECK-DAG: [[LOOP_FOREACH]] = distinct !{[[LOOP_FOREACH]], [[MUSTPROGRESS]]}
// CHECK-DAG: [[MUSTPROGRESS]] = !{!"llvm.loop.mustprogress"}
