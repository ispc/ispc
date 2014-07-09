/*
  Copyright (c) 2014, Evghenii Gaburov
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
void ispcSetMallocHeapLimit(size_t value)
{
}
void ispcSetStackLimit(size_t value)
{
}
unsigned long long ispcGetMallocHeapLimit()
{
  return -1;
}
unsigned long long ispcGetStackLimit()
{
  return -1;
}
void * ispcMemcpy(void *dest,  void *src,  size_t num)
{
  memcpy(dest, src, num);
  return dest;
}

#endif
