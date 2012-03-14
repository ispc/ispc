// Can't convert between types "uniform struct Point" and "soa<4> struct Point" with different SOA widths

struct Point { float a, b, c; };

void foo(soa<8> Point pts[]) {
    soa<4> Point x = pts[0];
}
