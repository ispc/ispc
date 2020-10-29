#include <ispc/expr.h>

#include <ispc/ast_node_visitor.h>

namespace ispc {

IntegerLiteral::IntegerLiteral(BinFormat, const char *text, std::size_t length) noexcept {
    for (std::size_t i = 0; i < length; i++) {
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

} // namespace ispc
