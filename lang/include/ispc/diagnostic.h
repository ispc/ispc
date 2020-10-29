#ifndef ISPC_LANG_INCLUDE_ISPC_DIAGNOSTIC_H
#define ISPC_LANG_INCLUDE_ISPC_DIAGNOSTIC_H

#include <iosfwd>

namespace ispc {

class Diagnostic {
  public:
    virtual ~Diagnostic() {}

    /** Prints the diagnostic to a stream,
     * in a human-readable format.
     *
     * @param stream The stream to print the diagnostic to.
     *               This would normally either be the standard
     *               error string or a string buffer stream.
     * */
    virtual void Print(std::ostream &stream) const = 0;
};

} // namespace ispc

#endif // ISPC_LANG_INCLUDE_ISPC_DIAGNOSTIC_H
