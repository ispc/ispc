#include "typeT.h"

static inline
uniform int gcd(uniform int a, uniform int b)
{
  while ( a != 0 )
  {
    uniform int c = a; 
    a = b%a;  
    b = c;
  }
  return b;
}

  static inline
int __rj(const int i, const uniform int joverb, const uniform int m, const uniform int b)
{
  return (i + joverb) % m;
}

  static inline
int __di(const int i, const uniform int j, const uniform int m, const uniform int n, const uniform int joverb)
{
  return (((i+joverb) % m) + j*m) % n;
}

  static inline
int __sj(const int i, const uniform int j, const uniform int m, const uniform int n, const int iovera)
{
  return (j + i*n - iovera) % m;
}

static inline 
void transpose_serial(uniform T A[], const uniform int m, const uniform int n)
{
  const uniform int tmpSize = max(m,n) * programCount;
  uniform T   * uniform tmp    = uniform new uniform T  [tmpSize];
  uniform int * uniform joverb = uniform new uniform int[n];
  uniform int * uniform iovera = uniform new uniform int[m];

  uniform T (*uniform tmp2D)[programCount] = (uniform T (*uniform)[programCount])tmp;

  const uniform int c = gcd(m,n);
  const uniform int a = m/c;
  const uniform int b = n/c;
  foreach (j = 0 ... n)
    joverb[j] = j/b;
  foreach (i = 0 ... m)
    iovera[i] = i/a;

  if (c > 1)
  {
    for (uniform int j = 0; j < n; j++)
    {
      const uniform int base = j*m;
      const uniform int __joverb = joverb[j];
      foreach (i = 0 ... m)
        tmp[i] = A[base + __rj(i,__joverb,m,b)];
      foreach (i = 0 ... m)
        A[base + i] = tmp[i];
    }
  }

  foreach (i = 0 ... m)
  {
    for (uniform int j = 0; j < n; j++)
      tmp2D[__di(i,j,m,n,joverb[j])][programIndex] = A[j*m + i];
    for (uniform int j = 0; j < n; j++)
      A[j*m + i] = tmp2D[j][programIndex];
  }

  for (uniform int j = 0; j < n; j++)
  {
    const uniform int base = j*m;
    foreach (i = 0 ... m)
      tmp[i] = A[base + __sj(i,j,m,n,iovera[i])];
    foreach (i = 0 ... m)
      A[base + i] = tmp[i];
  }

  delete iovera;
  delete joverb;
  delete tmp;
}

export 
void transpose(uniform T A[], const uniform int m, const uniform int n)
{
  transpose_serial(A, m, n);
}

