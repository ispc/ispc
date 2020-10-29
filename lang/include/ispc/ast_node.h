#ifndef ISPC_LANG_INCLUDE_ISPC_AST_NODE_H
#define ISPC_LANG_INCLUDE_ISPC_AST_NODE_H

namespace ispc {

class ASTNodeVisitor;

/** This is the base class of all components
 * to the AST.
 * */
class ASTNode {
  public:
    /** Just a stub. */
    virtual ~ASTNode() {}

    /** Used for accessing the derived type of the node.
     * Callers will use a class derived of @p visitor and
     * pass it as a parameter. The derived @ref ASTNode instance
     * will then call @ref ASTNodeVisitor::Visit with the appropriate
     * type info.
     * */
    virtual void Accept(ASTNodeVisitor &visitor) const = 0;
};

} // namespace ispc

#endif // ISPC_LANG_INCLUDE_ISPC_AST_NODE_H
