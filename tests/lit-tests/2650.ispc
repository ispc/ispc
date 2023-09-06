// RUN: %{ispc} --target=host --nostdlib --nowrap -h %t.h %s 2>&1
// RUN: FileCheck %s --input-file=%t.h
// RUN: %{cc} %t.h

// CHECK: struct uint32_t2 size;

struct Test {
    uint<2> size;
};

export void use(uniform Test * uniform test) {
}
