#include <iostream>
#include <cstdlib>
#include <cstdio>

struct Case
{
  int a; float b;
};

#if 0
void * operator new(size_t s) throw(std::bad_alloc)
{
  fprintf(stderr, "alloc %d bytes\n", (int)s);
  return (void*)0x123;
}
void operator delete(void *p) throw()
{
  fprintf(stderr, " free \n");
}
#else
inline void *malloc(size_t size)
{
  fprintf(stderr, "alloc %d bytes\n", (int)size);
  return (void*)0x123;
}
inline void free(void *ptr)
{
  fprintf(stderr, " free \n");
}
#endif

int main()
{
  Case *ptr = new Case[10];
  delete ptr;
  return 0;
}
