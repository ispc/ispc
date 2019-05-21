export void vecAdd(uniform float vin1[], uniform float vin2[], uniform float vout[],
                   uniform int count) {
    int index = programIndex;
    do {
        float v = vin1[index];
        if (v > 3.0f)
            v = v * v;
        vout[index] = v;
        index += programCount;
    } 
	while (index <= count);
}