#ifndef _CUDA_
#error "Something went wrong..."
#endif

void ispc_malloc(void **ptr, const size_t size)
{
  cudaMallocManaged(ptr, size);
}
void ispc_free(void *ptr)
{
  cudaFree(ptr);
}
void ispc_memset(void *ptr, int value, size_t size)
{
  cudaMemset(ptr, value, size);
}


