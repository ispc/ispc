//; RUN: not %{ispc} %s -o %t.o --nowrap --target=host 2>&1 | FileCheck %s
//; CHECK: Error: Initializer for global variable "g_final" must be a constant.
struct globStruct
  {
    int **struct_elem;
  };

extern uniform globStruct  g_globStruct;

int uniform * g_final = &g_globStruct.struct_elem[0][0];
