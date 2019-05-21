export void vecAdd(uniform int vin1[], uniform int vin2[], uniform int vout[],
                   uniform int count) {
    int index = programIndex;
    int v = vin1[index];
    if (v > 1) {
        v += vin2[index];
        if (v < 5) {
            v += 1000;
        }
        else {
            v += 10000;
        }
        }
    else {
        v += 100;
    }
    vout[index] = v;
}