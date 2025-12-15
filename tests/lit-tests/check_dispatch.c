// See check_dispatch.ispc for run recipe.

#include <stdio.h>

int detect_isa();

int main() {
  int isa = detect_isa();
  switch (isa) {
    case 1:
      printf("SSE2\n");
      break;
    case 2:
      printf("SSE4\n");
      break;
    case 3:
      printf("AVX1\n");
      break;
    case 4:
      printf("AVX2\n");
      break;
    case 5:
      printf("AVX2VNNI\n");
      break;
    case 7:
      printf("AVX512SKX\n");
      break;
    case 8:
      printf("AVX512ICL\n");
      break;
    case 9:
      printf("AVX512SPR\n");
      break;
    case 10:
      printf("AVX512GNR\n");
      break;
    case 11:
      printf("AVX10_2DMR\n");
      break;
    default:
      printf("Unknown ISA\n");
      break;
  }
  return 0;
}
