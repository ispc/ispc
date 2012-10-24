// Unable to apply SOA conversion to struct due to "uniform int32" member "x" with bound "uniform" 

struct Point { uniform int x; float a, b, c; };

void foo(soa<8> Point pts[]);
