// RUN: %{ispc} %s -MMM %t.dep -h %t.h > %t 2>&1
// RUN: FileCheck --input-file=%t %s --implicit-check-not "Writing dependencies"
int foo(int a) {
    return a;
}
