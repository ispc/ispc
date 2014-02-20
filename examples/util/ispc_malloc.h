#pragma once

extern void ispc_malloc(void **ptr, const size_t size);
extern void ispc_free(void *ptr);
extern void ispc_memset(void *ptr, int value, size_t size);
extern void ispcSetMallocHeapLimit(size_t value);
extern void ispcSetStackLimit(size_t value);
extern unsigned long long ispcGetMallocHeapLimit();
extern unsigned long long ispcGetStackLimit();
extern void * ispcMemcpy(void *dest,  void *src,  size_t num);
