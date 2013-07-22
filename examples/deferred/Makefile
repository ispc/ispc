
EXAMPLE=deferred_shading
CPP_SRC=common.cpp main.cpp dynamic_c.cpp dynamic_cilk.cpp
ISPC_SRC=kernels.ispc
ISPC_IA_TARGETS=sse2,sse4-x2,avx-x2
ISPC_ARM_TARGETS=neon
ISPC_FLAGS=--opt=fast-math

include ../common.mk
