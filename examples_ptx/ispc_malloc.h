#pragma once

#ifdef _CUDA_
extern void ispc_malloc(void **ptr, const size_t size);
extern void ispc_free(void *ptr);
extern void ispc_memset(void *ptr, int value, size_t size);
#else
#include <cstdlib>
static inline void ispc_malloc(void **ptr, const size_t size)
{
  *ptr = malloc(size);
}
static inline void ispc_free(void *ptr)
{
  free(ptr);
}
static inline void ispc_memset(void *ptr, int value, size_t size)
{
  memset(ptr, value, size);
}

#endif
