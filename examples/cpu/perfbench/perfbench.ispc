/*
  Copyright (c) 2012-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

export void xyzSumAOS(uniform float array[], uniform int count,
                      uniform float zeros[], uniform float result[]) {
    float xsum = 0, ysum = 0, zsum = 0;
    foreach (i = 0 ... count/3) {
        #pragma ignore warning(perf)
        float x = array[3*i];
        #pragma ignore warning(perf)
        float y = array[3*i+1];
        #pragma ignore warning(perf)
        float z = array[3*i+2];

        xsum += x;
        ysum += y;
        zsum += z;
    }
    result[0] = reduce_add(xsum);
    result[1] = reduce_add(ysum);
    result[2] = reduce_add(zsum);
}

export void xyzSumAOSStdlib(uniform float array[], uniform int count,
                            uniform float zeros[], uniform float result[]) {
    float xsum = 0, ysum = 0, zsum = 0;
    for (uniform int i = 0; i < 64*1024 /*count/3*/; i += programCount) {
        float x, y, z;
        aos_to_soa3(&array[3*i], &x, &y, &z);

        xsum += x;
        ysum += y;
        zsum += z;
    }
    result[0] = reduce_add(xsum);
    result[1] = reduce_add(ysum);
    result[2] = reduce_add(zsum);
}

export void xyzSumAOSNoCoalesce(uniform float array[], uniform int count,
                                uniform float zerosArray[], uniform float result[]) {
    int zeros = zerosArray[programIndex];
    float xsum = 0, ysum = 0, zsum = 0;
    foreach (i = 0 ... count/3) {
        #pragma ignore warning(perf)
        float x = array[3*i+zeros];
        #pragma ignore warning(perf)
        float y = array[3*i+1+zeros];
        #pragma ignore warning(perf)
        float z = array[3*i+2+zeros];

        xsum += x;
        ysum += y;
        zsum += z;
    }
    result[0] = reduce_add(xsum);
    result[1] = reduce_add(ysum);
    result[2] = reduce_add(zsum);
}

export void xyzSumSOA(uniform float array[], uniform int count,
                      uniform float zeros[], uniform float result[]) {
    float xsum = 0, ysum = 0, zsum = 0;
    uniform float * uniform ap = array;
    assert(programCount <= 8);

    for (uniform int i = 0; i < count/3; i += 8, ap += 24) {
        for (uniform int j = 0; j < 8; j += programCount) {
            float x = ap[j + programIndex];
            float y = ap[8 + j + programIndex];
            float z = ap[16 + j + programIndex];

            xsum += x;
            ysum += y;
            zsum += z;
        }
    }
    result[0] = reduce_add(xsum);
    result[1] = reduce_add(ysum);
    result[2] = reduce_add(zsum);
}

#pragma ignore warning
export void xyzSumVarying(varying float array[], uniform int count,
                          uniform float zeros[], uniform float result[]) {
    float xsum = 0, ysum = 0, zsum = 0;
    varying float * uniform ap = array;
    assert(programCount <= 8);

    if (programCount == 4) {
      for (uniform int i = 0; i < count/3; i += 8) {
        float x0 = ap[0];
        float y0 = ap[2];
        float z0 = ap[4];

        xsum += x0;
        ysum += y0;
        zsum += z0;

        float x1 = ap[1];
        float y1 = ap[3];
        float z1 = ap[5];

        xsum += x1;
        ysum += y1;
        zsum += z1;

        ap += 6;
      }
    }
    else {
      for (uniform int i = 0; i < count/3; i += 8) {
        // programCount == 8
        float x = ap[0];
        float y = ap[1];
        float z = ap[2];

        xsum += x;
        ysum += y;
        zsum += z;

        ap += 3;
      }
    }
    result[0] = reduce_add(xsum);
    result[1] = reduce_add(ysum);
    result[2] = reduce_add(zsum);
}

export void gathers(uniform float array[], uniform int count,
                    uniform float zeros[], uniform float result[]) {
    float sum = 0;
    int zero = zeros[programIndex];
    foreach (i = 0 ... count) {
        #pragma ignore warning(perf)
        sum += array[i + zero];
    }
    result[0] = reduce_add(sum);
}


export void loads(uniform float array[], uniform int count,
                  uniform float zeros[], uniform float result[]) {
    float sum = 0;
    foreach (i = 0 ... count) {
        #pragma ignore warning(perf)
        sum += array[i];
    }
    result[0] = reduce_add(sum);
}


export void scatters(uniform float array[], uniform int count,
                     uniform float zeros[], uniform float result[]) {
    int zero = zeros[programIndex];
    foreach (i = 0 ... count) {
        #pragma ignore warning(perf)
        array[i + zero] = zero;
    }
}


export void stores(uniform float array[], uniform int count,
                   uniform float zeros[], uniform float result[]) {
    int zero = zeros[programIndex];
    foreach (i = 0 ... count)
        array[i] = zero;
}

export void normalizeAOSNoCoalesce(uniform float array[], uniform int count,
                                   uniform float zeroArray[]) {
    float zeros = zeroArray[programIndex];
    foreach (i = 0 ... count/3) {
        #pragma ignore warning(perf)
        float x = array[3*i+zeros];
        #pragma ignore warning(perf)
        float y = array[3*i+1+zeros];
        #pragma ignore warning(perf)
        float z = array[3*i+2+zeros];

        float l2 = x*x + y*y + z*z;

        #pragma ignore warning(perf)
        array[3*i] /= l2;
        #pragma ignore warning(perf)
        array[3*i+1] /= l2;
        #pragma ignore warning(perf)
        array[3*i+2] /= l2;
    }
}

export void normalizeSOA(uniform float array[], uniform int count,
                         uniform float zeros[]) {
    foreach (i = 0 ... count/3) {
        #pragma ignore warning(perf)
        float x = array[3*i];
        #pragma ignore warning(perf)
        float y = array[3*i+1];
        #pragma ignore warning(perf)
        float z = array[3*i+2];

        float l2 = x*x + y*y + z*z;

        #pragma ignore warning(perf)
        array[3*i] /= l2;
        #pragma ignore warning(perf)
        array[3*i+1] /= l2;
        #pragma ignore warning(perf)
        array[3*i+2] /= l2;
    }
}
