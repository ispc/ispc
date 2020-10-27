#pragma once

class Module;

struct Globals;

namespace ispc {

Module *getModule();

Globals *getGlobals();

} // namespace ispc
