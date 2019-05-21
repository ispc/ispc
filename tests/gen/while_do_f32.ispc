export void vecAdd(uniform float vin1[], uniform float vin2[], uniform float vout[],
                   uniform int count) {
    int index = programIndex;
    while (index <= count) {    
        float v = vin1[index] + vin2[index];
        vout[index] = v;
        index += programCount;
    }
}