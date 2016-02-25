
EXAMPLE=deferred_shading
CPP_SRC=common.cpp main.cpp dynamic_c.cpp dynamic_cilk.cpp
ISPC_SRC=kernels.ispc
ISPC_IA_TARGETS=sse2-i32x4,sse4-i32x8,avx1-i32x16,avx2-i32x16,avx512knl-i32x16,avx512skx-i32x16
ISPC_ARM_TARGETS=neon
ISPC_FLAGS=--opt=fast-math

include ../common.mk
