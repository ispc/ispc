//; RUN: not %{ispc} --target=host %s --nowrap 2>&1 | FileCheck %s

//; CHECK: Error: SOA type parameter "soa_arg0" is illegal in an exported function.
//; CHECK: Error: SOA type parameter "soa_arg1" is illegal in an exported function.
//; CHECK: Error: SOA type parameter "soa_arg2" is illegal in an exported function.

typedef struct rgb {
    uint8 r, g, b;
} t_rgb;

export void foo(soa<4> t_rgb soa_arg0[], soa<4> t_rgb *soa_arg1, soa<4> t_rgb **soa_arg2) {}
