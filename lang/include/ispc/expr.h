#ifndef ISPC_LANG_INCLUDE_ISPC_EXPR_H
#define ISPC_LANG_INCLUDE_ISPC_EXPR_H

#include <ispc/ast_node.h>

#include <optional>
#include <string_view>

#include <cstdint>
#include <cstddef>

namespace ispc {

struct Token;

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
     * */
    IntegerLiteral(BinFormat, const std::string_view &text) noexcept;

    constexpr IntegerLiteral(std::uint64_t v) noexcept
        : value(v) {}

    void Accept(ASTNodeVisitor &) const override;

    constexpr std::uint64_t GetValue() const noexcept {
        return value;
    }
};

class StringLiteralImpl;

/** This enumerates the types of sequences that
 * can be found in a string literal.
 * */
enum class StringSequenceType {
    /** Normal text that is not preceded
     * with a backslash. */
    Normal,
    /** An unknown escape sequence, which
     * should be considered an error. */
    UnknownEscaped,
    /** A single character escape sequence. */
    EscapedChar,
    /** A series of octal digits after
     * the backslash of the escape sequence. */
    OctalDigits,
    /** An octal digit sequence that exceeded
     * the maximum value of a single character. */
    OctalDigitsOverflow,
    /** A series of hex digits after the
     * first backslash of the escape sequence. */
    HexDigits,
    /** A hex digit sequence that exceeded
     * the maximum allowable value for a single
     * character.
     * */
    HexDigitsOverflow
};

/** Converts a string sequence type to a
 * human readable string that describes the type.
 * */
constexpr const char *ToString(StringSequenceType type) noexcept;

/** Used to describe sequence of characters
 * within a string literal. This can either be
 * a span of normal characters, such as "abc",
 * or it can be an escaped sequence, such as "\x123".
 *
 * Malformed escaped sequences are also described
 * by this structure.
 * */
struct StringSequence final {

    /** The type of the sequence. */
    StringSequenceType type = StringSequenceType::Normal;

    /** A view of the original contents of the
     * string sequence.
     * */
    std::string_view text;

    /** A view of the string sequence after
     * the escape sequence (if there was one)
     * is resolved.
     * */
    std::string_view transformedText;
};

/** Made from @ref TokenType::StringLiteral but with all
 * escape sequences transformed.
 * */
class StringLiteral final : public PrimaryExpr {
    /** Contains the original token and transformed string. */
    StringLiteralImpl *self;
  public:
    /** Constructs a new string literal.
     * This function also parses the contents of @p token
     * to determine what escape sequences (if any) are found
     * in the body of the string literal.
     *
     * @param token The token that the string literal was
     *              found in. A copy of the original token
     *              text is made so that the lifetime of the
     *              node can be longer than that of the token.
     * */
    StringLiteral(const Token &token);

    /** Moves the string literal from one variable to another.
     *
     * @param other The string literal instance to be moved.
     *              After calling this function, the previous
     *              string literal instance should no longer
     *              be used.
     * */
    constexpr StringLiteral(StringLiteral &&other) noexcept
        : self(other.self) {
        other.self = nullptr;
    }

    /** Releases memory allocated by the string. */
    ~StringLiteral();

    void Accept(ASTNodeVisitor &visitor) const override;

    /** Gets an sequence from the string literal.
     *
     * @param index The index of the sequence to get.
     *              The return value of @ref StringLiteral::GetSequenceCount
     *              should be used to determine the upper limit
     *              of this parameter.
     *
     * @return The sequence at the specified index.
     *         In the case that the index is out of bounds,
     *         a default initialized sequence is returned
     *         instead.
     * */
    StringSequence GetSequence(std::size_t index) const noexcept;

    /** Gets the number of sequences in the string literal.
     * This function should be used before calling @ref StringLiteral::GetSequence.
     *
     * @return The number of sequences in the string literal.
     * */
    std::size_t GetSequenceCount() const noexcept;

    /** Accesses the transformed text of the
     * string literal.
     *
     * @return The transformed text of the string
     * literal, after all escape sequences are resolved.
     * */
    std::string_view GetText() const noexcept;

    /** Accesses the original token of the
     * string literal.
     *
     * @note The original token consists of quotation
     *       marks and all the original escape sequences.
     *       To get the text without quotations and transformed
     *       escaped sequences, use @ref StringLiteral::GetText.
     *
     * @return A copy of the original token.
     * */
    Token GetToken() const noexcept;
};

/* Implementation details below this point. */

constexpr const char *ToString(StringSequenceType type) noexcept {
    switch (type) {
    case StringSequenceType::Normal:
        return "Normal";
    case StringSequenceType::UnknownEscaped:
        return "Unknown Escape Sequence";
    case StringSequenceType::EscapedChar:
        return "Escaped Character";
    case StringSequenceType::OctalDigits:
        return "Octal Digits";
    case StringSequenceType::OctalDigitsOverflow:
        return "Octal Digits (Overflowed)";
    case StringSequenceType::HexDigits:
        return "Hex Digits";
    case StringSequenceType::HexDigitsOverflow:
        return "Hex Digits (Overflowed)";
    }
    return "";
}

} // namespace ispc

#endif // ISPC_LANG_INCLUDE_ISPC_EXPR_H
