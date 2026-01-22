// RUN: %ispc %s -o %t.o --target=host

// Regression test for allowing 32-bit enum values with the high bit set.

enum Flags {
    FLAG_HIGH_BIT = 0x80000000,
    FLAG_ALL = 0xffffffff
};

export uniform int use_enum_values() {
    uniform int a = (int)FLAG_HIGH_BIT;
    uniform int b = (int)FLAG_ALL;
    return a + b;
}
