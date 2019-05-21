export void vecAdd(uniform int vin1[], uniform int vin2[], uniform int vout[],
                   uniform int count) {
    int index = programIndex;
    while (index <= count) {    
        int v = vin1[index] + vin2[index];
        vout[index] = v;
        index += programCount;
    }
}