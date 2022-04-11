// This test is based on yarpgen test and checks that compiler does not crash during gather coalescing optimization.
/*
yarpgen version 2.0 (build c611daf on 2021:04:13)
Seed: 4096674170
Invocation: ../../build/yarpgen --std=ispc
*/

// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 --emit-spirv -o %t.spv
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-spirv -o %t.spv
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 --emit-spirv --opt=disable-xe-gather-coalescing -o %t.spv
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-spirv --opt=disable-xe-gather-coalescing -o %t.spv
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 --emit-spirv --opt=build-llvm-loads-on-xe-gather-coalescing -o %t.spv
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-spirv --opt=build-llvm-loads-on-xe-gather-coalescing -o %t.spv

// REQUIRES: XE_ENABLED

struct Parameters {
    uniform unsigned int8 var_1;
    uniform int8 var_2;
    uniform int32 var_3;
    uniform int32 var_5;
    uniform int8 var_7;
    uniform int32 var_11;
    uniform int8 var_16;
    uniform unsigned int16 var_18;
    uniform int16 var_19;
    uniform int32 var_20;
    uniform unsigned int64 arr_3[23][19];
    uniform unsigned int64 arr_19[23][11];
};

task void test(void *uniform _p) {

    Parameters *uniform p = (Parameters * uniform) _p;
    {
        if (((/* implicit */ uniform bool)((uniform int8)p->var_5))) {
            p->var_20 = ((/* implicit */ uniform int32)all((((/* implicit */ varying bool)((
                /* implicit */ uniform bool)((((/* implicit */ uniform bool)(uniform unsigned int8)93L))
                                                 ? (((/* implicit */ uniform int32)(uniform unsigned int16)15914L))
                                                 : (p->var_3)))))));
            /* LoopSeq 2 */
            foreach (i_0 = (((((/* implicit */ uniform int32)((/* implicit */ uniform unsigned int8)(
                                 (none((((/* implicit */ varying bool)((/* implicit */ uniform bool)p->var_19)))))
                                     ? (((/* implicit */ uniform int32)(uniform unsigned int16)49648L))
                                     : (p->var_11))))) -
                             (117L)) /*0*/)...(((((/* implicit */ uniform
                                                       int32)((/* implicit */ uniform unsigned int8)(
                                                    ~(((((/* implicit */ uniform bool)(uniform unsigned int8)132L))
                                                           ? (10621608389699517641ULL)
                                                           : (((/* implicit */ uniform unsigned int64)(
                                                                 (/* implicit */ uniform int32)p->var_7))))))))) -
                                                (54L)) /*0*/)) {
                for (uniform bool i_5 =
                         ((((/* implicit */ uniform int32)((/* implicit */ uniform bool)p->var_3))) - (1L)) /*0*/;
                     i_5 < (uniform bool)1L /*1*/;
                     i_5 += ((/* implicit */ uniform int32)((/* implicit */ uniform bool)(-(
                         ((((/* implicit */ uniform bool)p->var_18))
                              ? (((/* implicit */ uniform int32)p->var_16))
                              : (((/* implicit */ uniform int32)(uniform unsigned int8)246L))))))) /*1*/) {
                    p->arr_19[i_0][i_5] = ((/* implicit */ varying unsigned int64)p->arr_3[i_0][i_5]);
                }
            }
        }
    }
}
