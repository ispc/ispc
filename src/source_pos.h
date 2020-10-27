#pragma once

namespace llvm {

class DIFile;
class DINamespace;

} // namespace llvm

namespace ispc {

/** @brief Representation of a range of positions in a source file.

    This class represents a range of characters in a source file
    (e.g. those that span a token's definition), from starting line and
    column to ending line and column.  (These values are tracked by the
    lexing code).  Both lines and columns are counted starting from one.
 */
struct SourcePos final {

    SourcePos(const char *n = nullptr, int fl = 0, int fc = 0, int ll = 0, int lc = 0);

    const char *name;
    int first_line;
    int first_column;
    int last_line;
    int last_column;

    /** Prints the filename and line/column range to standard output. */
    void Print() const;

    /** Returns a LLVM DIFile object that represents the SourcePos's file */
    llvm::DIFile *GetDIFile() const;

    /** Returns a LLVM DINamespace object that represents 'ispc' namespace. */
    llvm::DINamespace *GetDINamespace() const;

    bool operator==(const SourcePos &p2) const;
};

/** Returns a SourcePos that encompasses the extent of both of the given
    extents. */
SourcePos Union(const SourcePos &p1, const SourcePos &p2);

} // namespace ispc
