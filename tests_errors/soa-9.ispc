// Can't convert from pointer to SOA type "soa<8> struct A \* uniform" to pointer to non-SOA type "void \* varying" 

struct A { float a, b; };

soa<8> A as[100];

void foo() {
    void *ptr = &as[0];
}
