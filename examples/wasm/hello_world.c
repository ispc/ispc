#include <stdint.h>
#include <malloc.h>
#include <stdio.h>

void hello_from_ispc(int *a, int *b, int *c, int N);

int main(int argc, char **argv) {
  printf("hello world from C!\n");
  int N = 1 << 6;
  int *data = (int *)malloc(N * 3 * sizeof(int));
  int *a = data;
  int *b = a + N;
  int *c = b + N;
  {
    int i;
    for (i = 0; i < N; i++)
      a[i] = i;
  }
  hello_from_ispc(a, b, c, N);
  free(data);
  return 0;
}