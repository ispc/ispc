// "uniform" qualifier and "soa<4>" qualifier can't both be used

struct F { float a, b, c; };

uniform soa<4> F farray[10];

