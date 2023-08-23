// RUN: %{ispc} --target=host --nostdlib --nowrap -o %t.o %s

typedef struct {
  int x;
  double y;
} MY_DATA;

export void foo_ispc(uniform MY_DATA records[])
{
  foreach(i=0 ... 10) {
    MY_DATA* d = records+i;
  }
}
