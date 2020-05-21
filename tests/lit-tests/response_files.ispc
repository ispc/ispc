//; RUN: %{ispc} @%S/1638.args  %s > %t
//; RUN: not %{ispc} %s @%S/missing_response_file.args -o %t.o --nowrap 2>&1 | FileCheck %s
//; RUN: not %{ispc} %s @%S/nested_response_files.args -o %t.o --nowrap 2>&1 | FileCheck %s


//; CHECK: Error: Multiple input files specified on command line:
//; CHECK: missing_response_file.args

void test(uniform float &Val) { Val = Val * -(Val * Val); }


