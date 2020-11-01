#ifndef ISPC_LANG_INCLUDE_ISPC_DIAGNOSTIC_CONSUMER_H
#define ISPC_LANG_INCLUDE_ISPC_DIAGNOSTIC_CONSUMER_H

namespace ispc {

class Diagnostic;

class DiagnosticConsumer {

  public:
    virtual ~DiagnosticConsumer() {}

    virtual void Consume(const Diagnostic &) = 0;
};

} // namespace ispc

#endif // ISPC_LANG_INCLUDE_ISPC_DIAGNOSTIC_CONSUMER_H
