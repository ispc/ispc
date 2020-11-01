#include <ispc/diagnostic_printer.h>

#include <ispc/diagnostic.h>

namespace ispc {

class DiagnosticPrinterImpl final {
    /** The stream being printed to. */
    std::ostream &stream;

  public:
    DiagnosticPrinterImpl(std::ostream &stream_) : stream(stream_) {}

    void Print(const Diagnostic &d) { d.Print(stream); }
};

DiagnosticPrinter::DiagnosticPrinter(std::ostream &stream) : self(new DiagnosticPrinterImpl(stream)) {}

DiagnosticPrinter::~DiagnosticPrinter() {
    delete self;
    self = nullptr;
}

void DiagnosticPrinter::Consume(const Diagnostic &d) {
    if (self)
        self->Print(d);
}

} // namespace ispc
