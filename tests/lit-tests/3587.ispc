// RUN: %{ispc} %s -O0 --target=sse4,avx1,avx2 --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=CHECK,SSE4
// RUN: %{ispc} %s -O0 --target=avx1,avx2 --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s
// REQUIRES: X86_ENABLED

// CHECK: @__system_best_isa = internal unnamed_addr global i32 -1, align 4
// CHECK: @__system_func_ptr_cache_bar = internal global ptr null
// CHECK: @__system_func_ptr_cache_foo = internal global ptr null

// CHECK: define internal i32 @__get_system_best_isa()

// CHECK:      define i32 @bar(i32 %0)
// CHECK-NEXT: entry:
// CHECK-NEXT:   %1 = load ptr, ptr @__system_func_ptr_cache_bar, align 8
// CHECK-NEXT:   %2 = icmp ne ptr %1, null
// CHECK-NEXT:   br i1 %2, label %skip_init, label %do_init
// CHECK:      skip_init:
// CHECK-NEXT:   %3 = load ptr, ptr @__system_func_ptr_cache_bar, align 8
// CHECK-NEXT:   %4 = call i32 %3(i32 %0)
// CHECK-NEXT:   ret i32 %4
// CHECK:      do_init:
// CHECK-NEXT:   %system_isa = call i32 @__get_system_best_isa()
// CHECK-NEXT:   %isa_ok = icmp sge i32 %system_isa, 5
// CHECK-NEXT:   br i1 %isa_ok, label %do_call, label %next_try
// CHECK:      do_call:
// CHECK-NEXT:   store ptr @bar_avx2, ptr @__system_func_ptr_cache_bar, align 8
// CHECK-NEXT:   br label %skip_init
// CHECK:      next_try:
// CHECK-NEXT:   %isa_ok1 = icmp sge i32 %system_isa, 3
// CHECK-NEXT:   br i1 %isa_ok1, label %do_call2, label %next_try3
// CHECK:      do_call2:
// CHECK-NEXT:   store ptr @bar_avx, ptr @__system_func_ptr_cache_bar, align 8
// CHECK-NEXT:   br label %skip_init
// CHECK:        next_try3:
// SSE4-NEXT:     %isa_ok4 = icmp sge i32 %system_isa, 2
// SSE4-NEXT:     br i1 %isa_ok4, label %do_call5, label %next_try6
// SSE4:        do_call5:
// SSE4-NEXT:     store ptr @bar_sse4, ptr @__system_func_ptr_cache_bar, align 8
// SSE4-NEXT:     br label %skip_init
// SSE4:        next_try6:
// CHECK-NEXT:   call void asm sideeffect "ud2", "~{dirflag},~{fpsr},~{flags}"()
// CHECK-NEXT:   unreachable

// CHECK:      define void @foo()
// CHECK-NEXT: entry:
// CHECK-NEXT:   %0 = load ptr, ptr @__system_func_ptr_cache_foo, align 8
// CHECK-NEXT:   %1 = icmp ne ptr %0, null
// CHECK-NEXT:   br i1 %1, label %skip_init, label %do_init
// CHECK:      skip_init:
// CHECK-NEXT:   %2 = load ptr, ptr @__system_func_ptr_cache_foo, align 8
// CHECK-NEXT:   call void %2()
// CHECK-NEXT:   ret void
// CHECK:      do_init:
// CHECK-NEXT:   %system_isa = call i32 @__get_system_best_isa()
// CHECK-NEXT:   %isa_ok = icmp sge i32 %system_isa, 5
// CHECK-NEXT:   br i1 %isa_ok, label %do_call, label %next_try
// CHECK:      do_call:
// CHECK-NEXT:   store ptr @foo_avx2, ptr @__system_func_ptr_cache_foo, align 8
// CHECK-NEXT:   br label %skip_init
// CHECK:      next_try:
// CHECK-NEXT:   %isa_ok1 = icmp sge i32 %system_isa, 3
// CHECK-NEXT:   br i1 %isa_ok1, label %do_call2, label %next_try3
// CHECK:      do_call2:
// CHECK-NEXT:   store ptr @foo_avx, ptr @__system_func_ptr_cache_foo, align 8
// CHECK-NEXT:   br label %skip_init
// CHECK:      next_try3:
// SSE4-NEXT:     %isa_ok4 = icmp sge i32 %system_isa, 2
// SSE4-NEXT:     br i1 %isa_ok4, label %do_call5, label %next_try6
// SSE4:        do_call5:
// SSE4-NEXT:     store ptr @foo_sse4, ptr @__system_func_ptr_cache_foo, align 8
// SSE4-NEXT:     br label %skip_init
// SSE4:        next_try6:
// CHECK-NEXT:   call void asm sideeffect "ud2", "~{dirflag},~{fpsr},~{flags}"()
// CHECK-NEXT:   unreachable

export void foo() {}

export uniform int bar(uniform int x) { return x; }
