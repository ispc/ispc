/*COextern "C" float f;*/
const uniform float f = 2.;

export void f_v(uniform float RET[]) {
    RET[programIndex] = f;
}

export void result(uniform float RET[]) { RET[programIndex] = 2.; }
