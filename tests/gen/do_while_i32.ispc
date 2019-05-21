export void vecAdd(uniform int vin1[], uniform int vin2[], uniform int vout[],
                   uniform int count) {
    int index = programIndex;
    do {
        int v = vin1[index];
        if (v > 3)
            v = v * v;
        vout[index] = v;
        index += programCount;
    } 
	while (index <= count);
}