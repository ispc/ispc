#include <ispc/expr.h>

#include <ispc/ast_node_visitor.h>
#include <ispc/token.h>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ispc {

IntegerLiteral::IntegerLiteral(BinFormat, const std::string_view &text) noexcept {
    for (std::size_t i = 0; i < text.size(); i++) {
        value <<= 1;
        if (text[i] == '1')
            value += 1;
        else if (text[i] != '0')
            break;
    }
}

void IntegerLiteral::Accept(ASTNodeVisitor &visitor) const {
    visitor.Visit(*this);
}

namespace {

struct StringSequenceImpl final {
    StringSequenceType type = StringSequenceType::Normal;
    std::string text;
    std::string transformedText;
};

/** Used for parsing the body of a string literal
 * and determining what escape sequences are found
 * in it.
 * */
class StringSequenceParser final {

    /** The string being parsed,
     * not including the beginning
     * and ending quotations. */
    std::string_view text;

    /** The index of the character
     * that the parser is currently at. */
    std::size_t index = 0;

  public:

    constexpr StringSequenceParser(const std::string_view &t) noexcept
        : text(t) {}

    constexpr bool IsDone() const noexcept {
        return index >= text.size();
    }

    std::optional<StringSequenceImpl> Parse() noexcept {

        if (IsDone())
            return std::nullopt;
        else if (Peek(0) != '\\')
            return CompleteNormalSequence();

        if (OutOfBounds(1))
            return std::nullopt;

        auto second = Peek(1);

        switch (second) {
        case 'a':
            return MakeChar('\a');
        case 'b':
            return MakeChar('\b');
        case 'f':
            return MakeChar('\f');
        case 'n':
            return MakeChar('\n');
        case 'r':
            return MakeChar('\r');
        case 't':
            return MakeChar('\t');
        case 'v':
            return MakeChar('\v');
        case '"':
        case '\'':
        case '\\':
            return MakeChar(second);
        case 'x':
            return CompleteHexDigits(2);
        }

        if (IsOctal(second))
            return CompleteOctalDigits(1);

        return Make(StringSequenceType::UnknownEscaped, 2, std::string(&second, 1));
    }

  protected:

    StringSequenceImpl CompleteNormalSequence() noexcept {

        std::size_t length = 0;

        while (!OutOfBounds(length) && (Peek(length) != '\\'))
            length++;

        return Make(StringSequenceType::Normal, length, std::string(text.data() + index, length));
    }

    StringSequenceImpl CompleteOctalDigits(std::size_t firstIndex) {

        while (Peek(firstIndex) == '0')
            firstIndex++;

        std::size_t length = firstIndex;

        int value = 0;

        auto overflowed = false;

        for (std::size_t i = firstIndex; i < Remaining(); i++) {

            if (!IsOctal(Peek(i)))
                break;

            value = (value * 8) + static_cast<int>(Peek(i) - '0');

            overflowed = overflowed || (value > 255);

            length++;
        }

        std::string transformedText;

        transformedText += static_cast<char>(value);

        auto type = overflowed ? StringSequenceType::OctalDigitsOverflow
                               : StringSequenceType::OctalDigits;

        return Make(type, length, std::move(transformedText));
    }

    StringSequenceImpl CompleteHexDigits(std::size_t firstIndex) {

        while (Peek(firstIndex) == '0')
            firstIndex++;

        std::size_t length = firstIndex;

        int value = 0;

        auto overflowed = false;

        for (std::size_t i = firstIndex; i < Remaining(); i++) {

            auto c = Peek(i);

            int digitValue = 0;

            if ((c >= '0') && (c <= '9'))
                digitValue = static_cast<int>(c - '0');
            else if ((c >= 'a') && (c <= 'f'))
                digitValue = static_cast<int>(c - 'a') + 10;
            else if ((c >= 'A') && (c <= 'F'))
                digitValue = static_cast<int>(c - 'A') + 10;
            else
                break;

            value = (value * 16) + digitValue;

            overflowed = overflowed || (value > 255);

            length++;
        }

        std::string transformedText;

        transformedText += static_cast<char>(value);

        auto type = overflowed ? StringSequenceType::HexDigitsOverflow
                               : StringSequenceType::HexDigits;

        return Make(type, length, std::move(transformedText));
    }

    StringSequenceImpl Make(StringSequenceType type, std::size_t length, std::string &&value) noexcept {

        if (index > text.size())
            index = text.size();

        if ((index + length) > text.size())
            length = text.size() - index;

        auto originalText = std::string(text.data() + index, length);

        StringSequenceImpl sequence {
            type,
            std::move(originalText),
            std::move(value)
        };

        index += length;

        return sequence;
    }

    StringSequenceImpl MakeChar(char c) noexcept {
        return Make(StringSequenceType::EscapedChar, 2, std::string(&c, 1));
    }

    static constexpr bool IsOctal(char c) noexcept {
        return (c >= '0') && (c <= '7');
    }

    constexpr std::size_t Remaining() const noexcept {
        return (index < text.size()) ? text.size() - index : 0;
    }

    constexpr bool OutOfBounds(std::size_t offset) const noexcept {
        return (index + offset) >= text.size();
    }

    constexpr char Peek(std::size_t offset) const noexcept {
        return OutOfBounds(offset) ? 0 : text[index + offset];
    }
};

} // namespace

class StringLiteralImpl final {

    std::string originalText;

    Token token;

    std::vector<StringSequenceImpl> sequences;

    std::string transformedText;

  public:

    StringLiteralImpl(const Token &t)
        : originalText(t.text, t.length),
          token { t.type, originalText.data(), originalText.size() } {

        StringSequenceParser parser(GetBodyOf(t.text, t.length));

        while (!parser.IsDone()) {

            auto seq = parser.Parse();
            if (!seq)
                break;

            sequences.emplace_back(std::move(*seq));
        }

        for (const auto &seq : sequences)
            transformedText += seq.transformedText;
    }

    StringSequence GetSequence(std::size_t index) const noexcept {

        if (index >= sequences.size())
            return StringSequence {};

        const auto &seq = sequences[index];

        return {
            seq.type,
            std::string_view(seq.text.data(), seq.text.size()),
            std::string_view(seq.transformedText.data(), seq.transformedText.size())
        };
    }

    std::size_t GetSequenceCount() const noexcept {
        return sequences.size();
    }

    std::string_view GetText() const noexcept {
        return std::string_view(transformedText.data(), transformedText.size());
    }

    Token GetToken() const noexcept {
        return token;
    }

  protected:

    static std::string_view GetBodyOf(const char *text, std::size_t length) {
        if ((length < 2) || (text[0] != '"') || (text[length - 1] != '"'))
            return std::string_view();
        else
            return std::string_view(text + 1, length - 2);
    }
};

StringLiteral::StringLiteral(const Token &token)
    : self(new StringLiteralImpl(token)) {}

StringLiteral::~StringLiteral() {
    delete self;
    self = nullptr;
}

void StringLiteral::Accept(ASTNodeVisitor &visitor) const {
    visitor.Visit(*this);
}

StringSequence StringLiteral::GetSequence(std::size_t index) const noexcept {

    if (!self)
        return StringSequence {};

    return self->GetSequence(index);
}

std::size_t StringLiteral::GetSequenceCount() const noexcept {

    if (!self)
        return 0;

    return self->GetSequenceCount();
}

std::string_view StringLiteral::GetText() const noexcept {

    if (!self)
        return "";

    return self->GetText();
}

Token StringLiteral::GetToken() const noexcept {

    if (!self)
        return Token { TokenType::StringLiteral, "", 0 };

    return self->GetToken();
}

} // namespace ispc
