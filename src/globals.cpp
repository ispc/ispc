#include "globals.h"

#include "ispc.h"

namespace ispc {

Globals *getGlobals() {
    return g;
}

Module *getModule() {
    return m;
}

} // namespace ispc
