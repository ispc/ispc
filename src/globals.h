#pragma once

struct Globals;

namespace ispc {

class Module;

Module *getModule();

Globals *getGlobals();

} // namespace ispc
