// RUN: %{ispc} %s --target=sse2-i32x4,sse4-i32x8,avx1-i32x8,avx2-i32x8,avx512skx-x16 --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s
// REQUIRES: X86_ENABLED

// CHECK: define i32 @add(i32 %0, i32 %1)
// CHECK: entry:
// CHECK:   %2 = load ptr, ptr @__system_func_ptr_cache, align 8
// CHECK:   %.not = icmp eq ptr %2, null
// CHECK:   br i1 %.not, label %do_init, label %skip_init
// CHECK: skip_init:
// CHECK:   %3 = phi ptr [ @add_sse2, %do_call11 ], [ @add_sse4, %do_call8 ], [ @add_avx, %do_call5 ], [ @add_avx2, %do_call2 ], [ @add_avx512skx, %do_call ], [ %2, %entry ]
// CHECK:   %4 = tail call i32 %3(i32 %0, i32 %1)
// CHECK:   ret i32 %4
// CHECK: do_init:
// CHECK:   %system_isa = tail call fastcc i32 @__get_system_best_isa()
// CHECK:   %isa_ok = icmp sgt i32 %system_isa, 7
// CHECK:   br i1 %isa_ok, label %do_call, label %next_try
// CHECK: do_call:
// CHECK:   store ptr @add_avx512skx, ptr @__system_func_ptr_cache, align 8
// CHECK:   br label %skip_init
// CHECK: next_try:
// CHECK:   %isa_ok1 = icmp sgt i32 %system_isa, 4
// CHECK:   br i1 %isa_ok1, label %do_call2, label %next_try3
// CHECK: do_call2:
// CHECK:   store ptr @add_avx2, ptr @__system_func_ptr_cache, align 8
// CHECK:   br label %skip_init
// CHECK: next_try3:
// CHECK:   %isa_ok4 = icmp sgt i32 %system_isa, 2
// CHECK:   br i1 %isa_ok4, label %do_call5, label %next_try6
// CHECK: do_call5:
// CHECK:   store ptr @add_avx, ptr @__system_func_ptr_cache, align 8
// CHECK:   br label %skip_init
// CHECK: next_try6:
// CHECK:   %isa_ok7 = icmp eq i32 %system_isa, 2
// CHECK:   br i1 %isa_ok7, label %do_call8, label %next_try9
// CHECK: do_call8:
// CHECK:   store ptr @add_sse4, ptr @__system_func_ptr_cache, align 8
// CHECK:   br label %skip_init
// CHECK: next_try9:
// CHECK:   %isa_ok10 = icmp sgt i32 %system_isa, -1
// CHECK:   br i1 %isa_ok10, label %do_call11, label %next_try12
// CHECK: do_call11:
// CHECK:   store ptr @add_sse2, ptr @__system_func_ptr_cache, align 8
// CHECK:   br label %skip_init
// CHECK: next_try12:
// CHECK:   tail call fastcc void @__terminate_now()
// CHECK:   unreachable

export uniform int add(uniform int x, uniform int y) {
    return x + y;
}
