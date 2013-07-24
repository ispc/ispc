// Assignment operator "\+=" is illegal with struct type

struct Point { float x, y, z; };

void foo() {
    Point a = {1,2,3}, b = {4,5,6};
    a += b;
}
