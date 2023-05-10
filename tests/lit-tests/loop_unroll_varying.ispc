// Test to check '#pragma unroll' functionality for different loop statements
//   with varying type induction variables.

// RUN: %{ispc} %s --target=host --nowrap -O0 --emit-llvm-text --no-discard-value-names --nostdlib -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECKO0
// RUN: %{ispc} %s --target=host --nowrap -O2 --emit-llvm-text --no-discard-value-names --nostdlib -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECKO2


#define NUM 1024


// CHECK_ALL-LABEL: define void @foo_for___

// CHECKO0:         for_loop:
// CHECKO0:         call void @goo_for___
// CHECKO0-NOT:     call void @goo_for___
// CHECKO0:         br label %{{[a-zA-Z_][a-zA-Z0-9_.]*}}
// CHECKO0:         }

// CHECKO2:         for_loop{{.*}}:
// CHECKO2:         call void @goo_for___
// CHECKO2:         for_loop{{.*}}:
// CHECKO2:         call void @goo_for___
// CHECKO2:         for_loop{{.*}}:
// CHECKO2:         call void @goo_for___
// CHECKO2-NOT:     call void @goo_for___
// CHECKO2:         br {{(i1)|(label)}} %{{[a-zA-Z_][a-zA-Z0-9_.]*}}
// CHECKO2:         }

extern void goo_for(int);
void foo_for() {
    #pragma unroll(3)
    for (int iter1 = 0; iter1 < NUM; iter1++) {
        goo_for(iter1);
    }
}


// CHECK_ALL-LABEL: define void @foo_for_unroll8___

// CHECKO0:         for_loop:
// CHECKO0:         call void @goo_for_unroll8___
// CHECKO0-NOT:     call void @goo_for_unroll8___
// CHECKO0:         br label %{{[a-zA-Z_][a-zA-Z0-9_.]*}}
// CHECKO0:         }

// CHECKO2:         for_loop{{.*}}:
// CHECKO2:         call void @goo_for_unroll8___
// CHECKO2:         for_loop{{.*}}:
// CHECKO2:         call void @goo_for_unroll8___
// CHECKO2:         for_loop{{.*}}:
// CHECKO2:         call void @goo_for_unroll8___
// CHECKO2:         for_loop{{.*}}:
// CHECKO2:         call void @goo_for_unroll8___
// CHECKO2:         for_loop{{.*}}:
// CHECKO2:         call void @goo_for_unroll8___
// CHECKO2:         for_loop{{.*}}:
// CHECKO2:         call void @goo_for_unroll8___
// CHECKO2:         for_loop{{.*}}:
// CHECKO2:         call void @goo_for_unroll8___
// CHECKO2:         for_loop{{.*}}:
// CHECKO2:         call void @goo_for_unroll8___
// CHECKO2-NOT:     call void @goo_for_unroll8___
// CHECKO2:         br {{(i1)|(label)}} %{{[a-zA-Z_][a-zA-Z0-9_.]*}}
// CHECKO2:         }

extern void goo_for_unroll8(int);
void foo_for_unroll8() {
    #pragma unroll(8)
    for (int iter1 = 0; iter1 < NUM; iter1++) {
        goo_for_unroll8(iter1);
    }
}


// CHECK_ALL-LABEL: define void @foo_for_packed___

// CHECKO0:         for_loop:
// CHECKO0:         call void @goo_for_packed___
// CHECKO0-NOT:     call void @goo_for_packed___
// CHECKO0:         br label %{{[a-zA-Z_][a-zA-Z0-9_.]*}}
// CHECKO0:         }

// CHECKO2:         for_loop{{.*}}:
// CHECKO2:         call void @goo_for_packed___
// CHECKO2:         for_loop{{.*}}:
// CHECKO2:         call void @goo_for_packed___
// CHECKO2:         for_loop{{.*}}:
// CHECKO2:         call void @goo_for_packed___
// CHECKO2:         for_loop{{.*}}:
// CHECKO2:         call void @goo_for_packed___
// CHECK02-NOT:     call void @goo_for_packed___
// CHECKO2:         br {{(i1)|(label)}} %{{[a-zA-Z_][a-zA-Z0-9_.]*}}
// CHECKO2:         }

extern void goo_for_packed(int);
void foo_for_packed() {
    #pragma unroll(4)
    for (int iter1 = programIndex; iter1 < NUM; iter1 += programCount) {
        goo_for_packed(iter1);
    }
}


// FIXME: The following test fails due to scalarization of the loop.

// COM: CHECK_ALL-LABEL: define void @foo_for_packed_nocall___

// COM: CHECKO0:         for_loop:
// COM: CHECKO0:         load <{{[0-9]*}} x i32>, <{{[0-9]*}} x i32>* %iter1{{[0-9]*}}
// COM: CHECKO0-NOT:     load <{{[0-9]*}} x i32>, <{{[0-9]*}} x i32>* %iter1{{[0-9]*}}
// COM: CHECKO0:         br label %{{[a-zA-Z_][a-zA-Z0-9_.]*}}
// COM: CHECKO0:         }

// COM: CHECKO2:         for_loop{{.*}}:
// COM: CHECKO2:         load <{{[0-9]*}} x i32>, <{{[0-9]*}} x i32>* %iter1{{[0-9]*}}
// COM: CHECKO2:         for_loop{{.*}}:
// COM: CHECKO2:         load <{{[0-9]*}} x i32>, <{{[0-9]*}} x i32>* %iter1{{[0-9]*}}
// COM: CHECKO2:         for_loop{{.*}}:
// COM: CHECKO2:         load <{{[0-9]*}} x i32>, <{{[0-9]*}} x i32>* %iter1{{[0-9]*}}
// COM: CHECKO2:         for_loop{{.*}}:
// COM: CHECKO2:         load <{{[0-9]*}} x i32>, <{{[0-9]*}} x i32>* %iter1{{[0-9]*}}
// COM: CHECKO2-NOT:     load <{{[0-9]*}} x i32>, <{{[0-9]*}} x i32>* %iter1{{[0-9]*}}
// COM: CHECKO2:         br {{(i1)|(label)}} %{{[a-zA-Z_][a-zA-Z0-9_.]*}}
// COM: CHECKO2:         }

//void foo_for_packed_nocall(uniform int _in[], uniform int _out[], const uniform int k) {
//    #pragma unroll(4)
//    for (int iter1 = programIndex; iter1 < NUM; iter1 += programCount) {
//        _out[iter1] = k * _in[iter1];
//    }
//}


// CHECK_ALL-LABEL: define <{{[0-9]*}} x i32> @foo_foreach___

// CHECKO0:         foreach_full_body:
// CHECKO0:         call <{{[0-9]*}} x i32> @goo_foreach___
// CHECKO0-NOT:     call <{{[0-9]*}} x i32> @goo_foreach___
// CHECKO0:         br label %{{[a-zA-Z_][a-zA-Z0-9_.]*}}
// CHECKO0:         }

// CHECKO2:         foreach_full_body:
// CHECKO2-COUNT-3: call <{{[0-9]*}} x i32> @goo_foreach___
// CHECKO2-NOT:     call <{{[0-9]*}} x i32> @goo_foreach___
// CHECKO2:         br {{(i1)|(label)}} %{{[a-zA-Z_][a-zA-Z0-9_.]*}}
// CHECKO2:         }

extern int goo_foreach(int);
int foo_foreach() {
    int res = 0;
    #pragma unroll(3)
    foreach (iter1 = 0 ... NUM) {
        res += goo_foreach(iter1);
    }
    return res;
}


// CHECK_ALL-LABEL: define <{{[0-9]*}} x i32> @foo_foreach_unroll8___

// CHECKO0:         foreach_full_body:
// CHECKO0:         call <{{[0-9]*}} x i32> @goo_foreach_unroll8___
// CHECKO0-NOT:     call <{{[0-9]*}} x i32> @goo_foreach_unroll8___
// CHECKO0:         br label %{{[a-zA-Z_][a-zA-Z0-9_.]*}}
// CHECKO0:         }

// CHECKO2:         foreach_full_body:
// CHECKO2-COUNT-8: call <{{[0-9]*}} x i32> @goo_foreach_unroll8___
// CHECKO2-NOT:     call <{{[0-9]*}} x i32> @goo_foreach_unroll8___
// CHECKO2:         br {{(i1)|(label)}} %{{[a-zA-Z_][a-zA-Z0-9_.]*}}
// CHECKO2:         }

extern int goo_foreach_unroll8(int);
int foo_foreach_unroll8() {
    int res = 0;
    #pragma unroll(8)
    foreach (iter1 = 0 ... NUM) {
        res += goo_foreach_unroll8(iter1);
    }
    return res;
}


// CHECK_ALL-LABEL: define <{{[0-9]*}} x i32> @foo_foreach_nocall___

// CHECKO0:         foreach_full_body:
// CHECKO0:         load <{{[0-9]*}} x i32>, {{.*}} %iter1{{[0-9]*}}
// CHECKO0-NOT:     load <{{[0-9]*}} x i32>, {{.*}} %iter1{{[0-9]*}}
// CHECKO0:         br label %{{[a-zA-Z_][a-zA-Z0-9_.]*}}
// CHECKO0:         }

// CHECKO2:         foreach_full_body:
// CHECKO2-COUNT-4: load <{{[0-9]*}} x i32>, {{.*}} %ptr
// CHECKO2-NOT:     load <{{[0-9]*}} x i32>, {{.*}} %ptr
// CHECKO2:         br {{(i1)|(label)}} %{{[a-zA-Z_][a-zA-Z0-9_.]*}}
// CHECKO2:         }

extern int goo_foreach_nocall(int);
int foo_foreach_nocall(uniform int _in[]) {
    int res = 0;
    #pragma unroll(4)
    foreach (iter1 = 0 ... NUM) {
        res += _in[iter1];
    }
    return res;
}


// CHECKO0: !{!"llvm.loop.unroll.count", i32 3}
// CHECKO0: !{!"llvm.loop.unroll.count", i32 8}
// CHECKO0: !{!"llvm.loop.unroll.count", i32 4}
