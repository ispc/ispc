//; RUN: not %{ispc} %s -o %t.o --nowrap --addressing=123 --x86-asm-syntax=amd --fast-math --fast-masked-vload --dwarf-version=15  --target=avx2-i32x8 --math-lib=fast2 -opt=make_me_happy -not-existing-opt %s 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_1
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 -I 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_2
//; RUN: not %{ispc} %s -o %t.o --nowrap --target 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_3
//; RUN: not %{ispc} %s -o 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_4
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 -h 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_5
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 -MMM 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_6
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 -MF 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_7
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 -MT 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_8
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 --dev-stub 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_9
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 --host-stub 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_10
//; RUN: not %{ispc} --nowrap --target=avx2-i32x8 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_11
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 --debug-phase= 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_12
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 --off-phase= 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_13
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 --debug-phase=12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_14
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 --off-phase=12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_15
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 --debug-phase=2000 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_16
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 --off-phase=2000 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_17
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 --arch=x64 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_18
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=sse3,avx2-i32x8,gost 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_19
//; RUN: not %{ispc} %s -o %t.o --nowrap --target sse3,avx2-i32x8,gost 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_20
//; RUN: %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 --arch=x86 --arch=x86_64 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_21
//; RUN: %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 --help
//; RUN: %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 --help-dev
//; RUN: %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 --support-matrix

//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 -not-existing-opt --quiet 2>&1 | FileCheck %s --allow-empty -check-prefix=CHECK_ERROR_QUIET

// REQUIRES: X86_ENABLED

//; CHECK_ERROR_1: Error: Addressing width "123" invalid -- only 32 and 64 are allowed.
//; CHECK_ERROR_1: Error: Invalid value for --x86-asm-syntax: "amd" -- only intel and att are allowed.
//; CHECK_ERROR_1: Error: --fast-math option has been renamed to --opt=fast-math!
//; CHECK_ERROR_1: Error: --fast-masked-vload option has been renamed to --opt=fast-masked-vload!
//; CHECK_ERROR_1: Error: Invalid value for DWARF version: "15" -- only 2, 3 and 4 are allowed.
//; CHECK_ERROR_1: Error: Unknown --math-lib= option "fast2".
//; CHECK_ERROR_1: Error: Unknown option "-opt=make_me_happy".
//; CHECK_ERROR_1: Error: Unknown option "-not-existing-opt".
//; CHECK_ERROR_1: Error: Multiple input files specified on command line:

//; CHECK_ERROR_2: No path specified after -I option.
//; CHECK_ERROR_3: No target specified after --target option.
//; CHECK_ERROR_4: No output file specified after -o option.
//; CHECK_ERROR_5: No header file name specified after -h option.
//; CHECK_ERROR_6: No output file name specified after -MMM option.
//; CHECK_ERROR_7: No output file name specified after -MF option.
//; CHECK_ERROR_8: No target name specified after -MT option.
//; CHECK_ERROR_9: No output file name specified after --dev-stub option.
//; CHECK_ERROR_10: No output file name specified after --host-stub option.
//; CHECK_ERROR_11: No input file were specified. To read text from stdin use "-" as file name.
//; CHECK_ERROR_12: Empty phase list.
//; CHECK_ERROR_13: Empty phase list.
//; CHECK_ERROR_14: Phase list is too long.
//; CHECK_ERROR_15: Phase list is too long.
//; CHECK_ERROR_16: Phases must be from 0 to 1000. 2000 is incorrect.
//; CHECK_ERROR_17: Phases must be from 0 to 1000. 2000 is incorrect.
//; CHECK_ERROR_18: Error: Unsupported value for --arch, supported values are:
//; CHECK_ERROR_19: Error: Incorrect targets: sse3,gost. Choices are:
//; CHECK_ERROR_20: Error: Incorrect targets: sse3,gost. Choices are:
//; CHECK_ERROR_21: Warning: Overwriting --arch=x86 with --arch=x86-64

//; CHECK_ERROR_QUIET-NOT: Error

uniform int i;
