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


#if 0
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
#endif

static uniform int * uniform joverb = NULL;
static uniform int * uniform iovera = NULL;
static uniform int a,b,c;

static 
void transpose_init(const uniform int m, const uniform int n, const uniform int nTask)
{
  joverb = uniform new uniform int[n];
  iovera = uniform new uniform int[m];

  c = gcd(m,n);
  a = m/c;
  b = n/c;
  foreach (j = 0 ... n)
    joverb[j] = j/b;
  foreach (i = 0 ... m)
    iovera[i] = i/a;
}

static 
void transpose_finalize()
{
  delete iovera;
  delete joverb;
}

task 
void transpose_step1(uniform T A[], const uniform int m, const uniform int n)
{
  const uniform int n_per_task = (n + taskCount - 1)/taskCount;
  const uniform int nibeg = taskIndex * n_per_task;
  const uniform int niend = min(nibeg + n_per_task, n);

  uniform T * uniform tmp = uniform new uniform T[m];

  for (uniform int j = nibeg; j < niend; j++)
  {
    const uniform int base = j*m;
    const uniform int __joverb = joverb[j];
    foreach (i = 0 ... m)
      tmp[i] = A[base + __rj(i,__joverb,m,b)];
    foreach (i = 0 ... m)
      A[base + i] = tmp[i];
  }

  delete tmp;
}

task
void transpose_step2(uniform T A[], const uniform int m, const uniform int n)
{
  const uniform int m_per_task = (m + taskCount - 1)/taskCount;
  const uniform int mibeg = taskIndex * m_per_task;
  const uniform int miend = min(mibeg + m_per_task, m);
  
  uniform  T * uniform tmp = uniform new uniform T[n*programCount];

  uniform T (*uniform tmp2D)[programCount] = (uniform T (*uniform)[programCount])tmp;
  foreach (i = mibeg ... miend)
  {
    for (uniform int j = 0; j < n; j++)
      tmp2D[__di(i,j,m,n,joverb[j])][programIndex] = A[j*m + i];
    for (uniform int j = 0; j < n; j++)
      A[j*m + i] = tmp2D[j][programIndex];
  }

  delete tmp;
}

task
void transpose_step3(uniform T A[], const uniform int m, const uniform int n)
{
  const uniform int n_per_task = (n + taskCount - 1)/taskCount;
  const uniform int nibeg = taskIndex * n_per_task;
  const uniform int niend = min(nibeg + n_per_task, n);
  
  uniform  T * uniform tmp = uniform new uniform T[m];

  for (uniform int j = nibeg; j < niend; j++)
  {
    const uniform int base = j*m;
    foreach (i = 0 ... m)
      tmp[i] = A[base + __sj(i,j,m,n,iovera[i])];
    foreach (i = 0 ... m)
      A[base + i] = tmp[i];
  }

  delete tmp;
}

export 
void transpose(uniform T A[], const uniform int m, const uniform int n)
{
#if 0
  transpose_serial(A, m, n);
#else
  const uniform int nTask = 32*8; //num_cores()*4;
  transpose_init(m,n,nTask);

  launch [nTask] transpose_step1(A, m, n);
  sync;
  
  launch [nTask] transpose_step2(A, m, n);
  sync;

  launch [nTask] transpose_step3(A, m, n);
  sync;

  transpose_finalize();
#endif
  sync;
}

