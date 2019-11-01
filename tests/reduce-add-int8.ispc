export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
  int8 db = b-4;
  int8 iv = programIndex + db;
  int m = reduce_add(iv);
  RET[programIndex] = m;
}

export void result(uniform float RET[]) { 
    uniform int x = 0;
    for (uniform int i = 1; i <= programCount; ++i)
        x += i;
    RET[programIndex] = x;
}

