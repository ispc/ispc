
export uniform int width() { return programCount; }


task void f_f_task(uniform float RET[], uniform float aFOO[]) {
    uniform float val[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        val[i] = 0;

    foreach (i = 0 ... programCount)
        val[i] += aFOO[programCount*taskIndex + i] - 1;

    uniform float sum = 0;
    for (uniform int i = 0; i < programCount; ++i)
        sum += val[i];

    if (programIndex < 32/4)
      RET[programCount/4*taskIndex + programIndex] = sum;
}

export void f_f(uniform float RET[], uniform float aFOO[]) 
{
  launch[4] f_f_task(RET,  aFOO);
}
task void result_task(uniform float RET[])
{
  const uniform float ret = reduce_add(programIndex + programCount*taskIndex);
  if (programIndex < 32/4)
    RET[programCount/4*taskIndex + programIndex] = ret;
}

export void result(uniform float RET[]) {
  launch[4] result_task(RET);
}
