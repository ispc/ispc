// RUN: %{ispc} --target=host %s > %t 2>&1
typedef bool<2> bool2;
void MakeBool2(bool2 arg1) { return; }
