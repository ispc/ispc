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
void ispcSetMallocHeapLimit(size_t value)
{
  cudaDeviceSetLimit(cudaLimitMallocHeapSize,value);
}
void ispcSetStackLimit(size_t value)
{
  cudaDeviceSetLimit(cudaLimitStackSize,value);
}
unsigned long long ispcGetMallocHeapLimit()
{
  size_t value;
  cudaDeviceGetLimit(&value, cudaLimitMallocHeapSize);
  return value;
}
unsigned long long ispcGetStackLimit()
{
  size_t value;
  cudaDeviceGetLimit(&value, cudaLimitStackSize);
  return value;
}
void * ispcMemcpy(void *dest,  void *src,  size_t num)
{
  cudaMemcpy(dest, src, num, cudaMemcpyDefault);
  return dest;
}


