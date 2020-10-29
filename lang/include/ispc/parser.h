#ifndef ISPC_LANG_INCLUDE_ISPC_PARSER_H
#define ISPC_LANG_INCLUDE_ISPC_PARSER_H

#include <ispc/token_consumer.h>

#include <memory>

namespace ispc {

class ASTNodeConsumer;
class DiagnosticConsumer;
class ParserImpl;

class Parser final : public TokenConsumer {
    /** Contains the implementation data. */
    ParserImpl *self = nullptr;
public:

    constexpr Parser() noexcept {}

    constexpr Parser(Parser &&other) noexcept : self(other.self) {
        self = other.self;
    }

    ~Parser();

    /** Adds an consumer to pass AST nodes to
     * as they are found by the parser.
     *
     * @param consumer A pointer to the consumer to add.
     *                 After calling this function, the parser
     *                 takes ownership of the pointer.
     * */
    void AddASTNodeConsumer(std::unique_ptr<ASTNodeConsumer> &&consumer);

    /** Adds a diagnostic consumer to handle
     * syntax errors found by the parser.
     *
     * @param consumer A pointer to the diagnostic
     *                 consumer to add. The parser takes
     *                 ownership of the pointer after calling
     *                 this function.
     * */
    void AddDiagnosticConsumer(std::unique_ptr<DiagnosticConsumer> &&consumer);

    /** Consumes a token produced
     * by an instance of @ref Scanner.
     *
     * May trigger the production of an AST node.
     * */
    void Consume(const Token &) override;

    /** Indicates to the parser that no more tokens
     * will be passed to @ref Parser::Consume and that
     * the end of the file has been reached.
     * */
    void Finish();

    /** This function indicates if the parser
     * is expecting more data.
     * */
    bool NeedsMore() const noexcept;
};

} // namespace ispc

#endif // ISPC_LANG_INCLUDE_ISPC_PARSER_H
