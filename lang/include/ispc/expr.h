#ifndef ISPC_LANG_INCLUDE_ISPC_EXPR_H
#define ISPC_LANG_INCLUDE_ISPC_EXPR_H

#include <ispc/ast_node.h>

#include <cstdint>
#include <cstddef>

namespace ispc {

class ASTNodeVisitor;

class Expr : public ASTNode {
  public:
    virtual ~Expr() {}
};

class PrimaryExpr : public Expr {
  public:
    virtual ~PrimaryExpr() {}
};

class BinFormat final {};
class DecFormat final {};
class HexFormat final {};

class IntegerLiteral final : public PrimaryExpr {

    std::uint64_t value = 0;

  public:

    /** Parses an integer literal value from a binary sequence.
     *
     * @note This function ignores errors in the sequence, since they are
     * expected to be caught by the token scanner.
     *
     * @param text The text containing the binary sequence.
     *             This should not include the "0b" prefix.
     *             The string does not have to be null-terminated.
     *
     * @param length The number of bytes in @p text.
     * */
    IntegerLiteral(BinFormat, const char *text, std::size_t length) noexcept;

    constexpr IntegerLiteral(std::uint64_t v) noexcept
        : value(v) {}

    void Accept(ASTNodeVisitor &) const override;

    constexpr std::uint64_t GetValue() const noexcept {
        return value;
    }
};

} // namespace ispc

#endif // ISPC_LANG_INCLUDE_ISPC_EXPR_H
