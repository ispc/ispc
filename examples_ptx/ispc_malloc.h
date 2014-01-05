#pragma once

extern void ispc_malloc(void **ptr, const size_t size);
extern void ispc_free(void *ptr);
extern void ispc_memset(void *ptr, int value, size_t size);
