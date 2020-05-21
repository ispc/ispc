//; RUN: not %{ispc} %s --nowrap --target=host 2>&1 | FileCheck %s -check-prefix=CHECK_ALL_ERR
//; RUN: not %{ispc} %s --nowrap --error-limit=2 --target=host 2>&1 | FileCheck %s -check-prefix=CHECK_2_ERR
//; RUN: not %{ispc} %s --nowrap --error-limit=0 --target=host 2>&1 | FileCheck %s --implicit-check-not "ERROR:"
//; RUN: not %{ispc} %s --nowrap --error-limit=-4 --target=host 2>&1 | FileCheck %s -check-prefix=CHECK_INVALID

//; CHECK_INVALID: Error: Invalid value for --error-limit: "-4" -- value cannot be a negative number.

const int r = 7;

//; CHECK_ALL_ERR: 12:18: Error: Initializer for global variable "n" must be a constant.
//; CHECK_2_ERR: 12:18: Error: Initializer for global variable "n" must be a constant.
uniform int n = {1, 3};

//; CHECK_ALL_ERR: 16:19: Error: Can't convert from type "const varying int32" to type "uniform float" for initializer.
//; CHECK_2_ERR: 16:19: Error: Can't convert from type "const varying int32" to type "uniform float" for initializer.
uniform float p = r;

//; CHECK_ALL_ERR: 20:1: Error: syntax error, unexpected identifier.
//; CHECK_2_ERR-NOT: 20:1: Error: syntax error, unexpected identifier.
r = 1;
