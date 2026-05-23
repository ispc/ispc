#include "test_static.isph"

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
  int8 db = b - 4;
  uint8 iv = (uint8)((programIndex + db - 19) % 7);
  uint8 m = reduce_add(iv);
  RET[programIndex] = m;
}

task void result(uniform float RET[]) {
    uniform uint8 x = 0;
    for (uniform int i = 1; i <= programCount; ++i){
        x += (uint8)((i - 19) % 7);
    }
    RET[programIndex] = x;
}
