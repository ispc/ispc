// rule: skip on arch=genx32
// rule: skip on arch=genx64

#define THREADS 4

task void f_f_task(uniform float RET[], uniform float aFOO[]) {
    uniform float val[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        val[i] = 0;

    foreach (i = 0 ... programCount)
        val[i] += aFOO[programCount*taskIndex + i] - 1;

    uniform float sum = 0;
    for (uniform int i = 0; i < programCount; ++i)
        sum += val[i];

    if (programIndex < programCount/THREADS)
        RET[programCount/THREADS*taskIndex + programIndex] = sum;
}

export void f_f(uniform float RET[], uniform float aFOO[])
{
    launch[THREADS] f_f_task(RET,  aFOO);
}
task void result_task(uniform float RET[])
{
    const uniform float ret = reduce_add(programIndex + programCount*taskIndex);
    if (programIndex < programCount/THREADS)
        RET[programCount/THREADS*taskIndex + programIndex] = ret;
}

export void result(uniform float RET[]) {
    launch[THREADS] result_task(RET);
}
