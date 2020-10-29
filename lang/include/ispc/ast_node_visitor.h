#ifndef ISPC_LANG_INCLUDE_ISPC_AST_NODE_VISITOR_H
#define ISPC_LANG_INCLUDE_ISPC_AST_NODE_VISITOR_H

namespace ispc {

class IntegerLiteral;
class Identifier;

class ASTNodeVisitor {

  public:

    virtual ~ASTNodeVisitor() {}

    virtual void Visit(const IntegerLiteral &) = 0;

    virtual void Visit(const Identifier &) = 0;
};

} // namespace ispc

#endif // ISPC_LANG_INCLUDE_ISPC_AST_NODE_VISITOR_H
