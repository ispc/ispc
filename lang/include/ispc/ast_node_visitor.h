#ifndef ISPC_LANG_INCLUDE_ISPC_AST_NODE_VISITOR_H
#define ISPC_LANG_INCLUDE_ISPC_AST_NODE_VISITOR_H

namespace ispc {

class IntegerLiteral;
class Identifier;
class StringLiteral;

/** This class is used for accessing the
 * derived type of an instance of @ref ASTNode.
 *
 * Passing an instance of a  class derived
 * from @ref ASTNodeVisitor to the function
 * @ref ASTNode::Accept will trigger one of
 * the "Visit" functions for the appropriate
 * type.
 * */
class ASTNodeVisitor {

  public:
    virtual ~ASTNodeVisitor() {}

    virtual void Visit(const IntegerLiteral &) = 0;

    virtual void Visit(const Identifier &) = 0;

    virtual void Visit(const StringLiteral &) = 0;
};

} // namespace ispc

#endif // ISPC_LANG_INCLUDE_ISPC_AST_NODE_VISITOR_H
