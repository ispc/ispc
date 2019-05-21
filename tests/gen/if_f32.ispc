export void vecAdd(uniform float vin1[], uniform float vin2[], uniform float vout[],
                   uniform int count) {
    int index = programIndex;
    float v = vin1[index];
    if (v > 1.0f) {
        v += vin2[index];
        if (v < 5.0f) {
            v += 1000.5f;
        }
        else {
            v += 10000.5f;
        }
        }
    else {
        v += 100.5f;
    }
    vout[index] = v;
}