// Test to check '#pragma nounroll' functionality for different loop statemnts.

// RUN: %{ispc} %s --target=host --nowrap -O0 --emit-llvm-text --no-discard-value-names --nostdlib -o - | FileCheck %s -check-prefix=CHECKO0
// RUN: %{ispc} %s --target=host --nowrap -O2 --emit-llvm-text --no-discard-value-names --nostdlib -o - | FileCheck %s -check-prefix=CHECKO2

// CHECKO0: define void @foo_for___
// CHECKO0: br label %{{[a-zA-Z_][a-zA-Z0-9_]*}}, !llvm.loop

// CHECKO2: define void @foo_for___
// CHECKO2: call void @goo_for___uni
// CHECKO2-NOT: call void @goo_for___uni

extern void goo_for(uniform int);
void foo_for() {
#pragma nounroll
    for (uniform int iter1 = 0; iter1 < 45; iter1++) {
        goo_for(iter1);
    }   
}

// CHECKO0: define void @foo_do___
// CHECKO0: br i1 %{{[a-zA-Z_][a-zA-Z0-9_]*}}, label %{{[a-zA-Z_][a-zA-Z0-9_]*}}, label %{{[a-zA-Z_][a-zA-Z0-9_]*}}, !llvm.loop

// CHECKO2: define void @foo_do___
// CHECKO2: call void @goo_do___uni
// CHECKO2-NOT: call void @goo_do___uni

extern void goo_do(uniform int);
void foo_do() {
    uniform int iter1 = 0;
#pragma nounroll
    do {
        goo_do(iter1);
        iter1++;
    } while(iter1 < 45);
}

// CHECKO0: !{!"llvm.loop.unroll.disable"}
