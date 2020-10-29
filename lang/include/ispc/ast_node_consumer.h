#ifndef ISPC_LANG_INCLUDE_ISPC_AST_NODE_CONSUMER_H
#define ISPC_LANG_INCLUDE_ISPC_AST_NODE_CONSUMER_H

namespace ispc {

class ASTNode;

/** Used for consuming AST nodes
 * when they are produced by the parser.
 * */
class ASTNodeConsumer {
  public:
    /** Just a stub. */
    virtual ~ASTNodeConsumer() {}

    /** Called whenever an instance of @ref Parser
     * successfully parses a node.
     *
     * @note Very little validation is done by the parser.
     *       There is a change that the node passed to this
     *       function still contains an error.
     * */
    virtual void Consume(const ASTNode &) = 0;
};

} // namespace ispc

#endif // ISPC_LANG_INCLUDE_ISPC_AST_NODE_CONSUMER_H
