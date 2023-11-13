// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: syntax error, unexpected '('

int * func(int a) {
    return new int[a](10);
}
