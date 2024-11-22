// RUN: %{ispc} %s --nowrap --target=host --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s

#define unmangled __attribute__((unmangled))
#define ATTRS unmasked unmangled

#if TARGET_WIDTH == 2
#define DOUBLE_WIDTH 4
#elif TARGET_WIDTH == 4
#define DOUBLE_WIDTH 8
#elif TARGET_WIDTH == 8
#define DOUBLE_WIDTH 16
#elif TARGET_WIDTH == 16
#define DOUBLE_WIDTH 32
#elif TARGET_WIDTH == 32
#define DOUBLE_WIDTH 64
#elif TARGET_WIDTH == 64
#define DOUBLE_WIDTH 128
#else
#error Unknown value of TARGET_WIDTH
#endif

// CHECK: @double_width_vec = local_unnamed_addr global <[[DOUBLE_WIDTH:.*]] x i32> zeroinitializer
uniform int<DOUBLE_WIDTH> double_width_vec;

// CHECK: define void @mask_type___(<[[WIDTH:.*]] x [[MASK:.*]]> %__mask)
void mask_type() {}

// CHECK-LABEL: @atomicrmw
// CHECK-DAG:   %calltmp = tail call double @llvm.ispc.atomicrmw.xchg.seq_cst.f64([[PTR:(ptr|i8\*)]] %ptr, double %val)
ATTRS uniform double atomicrmw(uniform int8 *uniform ptr, uniform double val) {
    return @llvm.ispc.atomicrmw.xchg.seq_cst(ptr, val);
}

// CHECK-LABEL: @bitcast
// CHECK-DAG:   %calltmp = tail call half @llvm.ispc.bitcast.i16.f16(i16 %x, half 0xH0000)
ATTRS uniform float16 bitcast(uniform int16 x) {
    return @llvm.ispc.bitcast(x, (uniform float16)0);
}

// CHECK-LABEL: @concat
// CHECK-DAG: %calltmp = tail call <[[DOUBLE_WIDTH]] x i8> @llvm.ispc.concat.v[[DOUBLE_WIDTH]]i8.v[[WIDTH]]i8(<[[WIDTH]] x i8> %v0, <[[WIDTH]] x i8> %v1)
ATTRS uniform int8<DOUBLE_WIDTH> concat(int8 v0, int8 v1) { return @llvm.ispc.concat(v0, v1); }

// CHECK-LABEL: @cmpxchg
// CHECK-DAG:   %calltmp = tail call i32 @llvm.ispc.cmpxchg.seq_cst.seq_cst.i32([[PTR]] %ptr, i32 %cmp, i32 %val)
ATTRS uniform int32 cmpxchg(uniform int8 *uniform ptr, uniform int32 cmp, uniform int32 val) {
    return @llvm.ispc.cmpxchg.seq_cst.seq_cst(ptr, cmp, val);
}

// CHECK-LABEL: @f_extract
// CHECK-DAG:   %calltmp = tail call i64 @llvm.ispc.extract.v[[WIDTH]]i64(<[[WIDTH]] x i64> %v, i32 %i)
ATTRS uniform uint64 f_extract(uint64 v, uniform int32 i) { return @llvm.ispc.extract(v, i); }

// CHECK-LABEL: @fence
// CHECK-DAG:   tail call void @llvm.ispc.fence.seq_cst()
ATTRS void fence() { @llvm.ispc.fence.seq_cst(); }

// CHECK-LABEL: @f_insert
// CHECK-DAG:   %calltmp = tail call <[[WIDTH]] x double> @llvm.ispc.insert.v[[WIDTH]]f64(<[[WIDTH]] x double> %v, i32 %i, double %val)
ATTRS double f_insert(double v, uniform int32 i, uniform double val) { return @llvm.ispc.insert(v, i, val); }

// CHECK-LABEL: @packmask
// CHECK-DAG:   %calltmp = tail call i64 @llvm.ispc.packmask.v[[WIDTH]][[MASK]](<[[WIDTH]] x [[MASK]]> %__mask)
unmangled uniform uint64 packmask() {
    return @llvm.ispc.packmask(__mask);
}

// CHECK-LABEL: @f_select
// CHECK-DAG:   %calltmp = tail call <[[WIDTH]] x float> @llvm.ispc.select.v[[WIDTH]]f32(<[[WIDTH]] x [[MASK]]> %{{.*}}, <[[WIDTH]] x float> %val, <[[WIDTH]] x float> %elt)
ATTRS float f_select(bool cond, float val, float elt) { return @llvm.ispc.select(cond, val, elt); }

// CHECK-LABEL: @streaming_load
// CHECK-DAG:   %calltmp = tail call i8 @llvm.ispc.stream_load.i8([[PTR]] %ptr, i8 %dummy)
ATTRS uniform int8 streaming_load(uniform int8 *uniform ptr, uniform int8 dummy) {
    return @llvm.ispc.stream_load(ptr, dummy);
}

// CHECK-LABEL: @streaming_store
// CHECK-DAG:   tail call void @llvm.ispc.stream_store.f16([[PTR]] %ptr, half %val)
ATTRS void streaming_store(uniform int8 *uniform ptr, uniform float16 val) {
    @llvm.ispc.stream_store(ptr, val);
}
