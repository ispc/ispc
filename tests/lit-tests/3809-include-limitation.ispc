// XFAIL: *
// RUN: %{ispc} %s --target=host --nostdlib --nowrap -o %t.o

// FIXME: Included files are not normalized yet.
#include "3809-include-limitation.isph"
