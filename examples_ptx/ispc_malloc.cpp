#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include "ispc_malloc.h"

#ifdef _CUDA_

void * operator new(size_t size) throw(std::bad_alloc)
{
  void *ptr;
  ispc_malloc(&ptr, size);
  return ptr;
}
void operator delete(void *ptr) throw()
{
  ispc_free(ptr);
}

#else

void ispc_malloc(void **ptr, const size_t size)
{
  *ptr = malloc(size);
}
void ispc_free(void *ptr)
{
  free(ptr);
}
void ispc_memset(void *ptr, int value, size_t size)
{
  memset(ptr, value, size);
}

#endif
