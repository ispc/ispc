#include "source_pos.h"

#include "globals.h"
#include "module.h"
#include "util.h"

#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Module.h>

namespace ispc {

SourcePos::SourcePos(const char *n, int fl, int fc, int ll, int lc) {
    name = n;
    if (name == NULL) {
        if (getModule() != nullptr)
            name = getModule()->GetLLVMModule()->getModuleIdentifier().c_str();
        else
            name = "(unknown)";
    }
    first_line = fl;
    first_column = fc;
    last_line = ll != 0 ? ll : fl;
    last_column = lc != 0 ? lc : fc;
}

llvm::DIFile *
// llvm::MDFile*
SourcePos::GetDIFile() const {
    std::string directory, filename;
    GetDirectoryAndFileName(g->currentDirectory, name, &directory, &filename);
    llvm::DIFile *ret = m->GetDIBuilder()->createFile(filename, directory);
    return ret;
}

llvm::DINamespace *SourcePos::GetDINamespace() const {
    llvm::DIScope *discope = GetDIFile();
    llvm::DINamespace *ret = m->GetDIBuilder()->createNameSpace(discope, "ispc", true);
    return ret;
}

void SourcePos::Print() const {
    printf(" @ [%s:%d.%d - %d.%d] ", name, first_line, first_column, last_line, last_column);
}

bool SourcePos::operator==(const SourcePos &p2) const {
    return (!strcmp(name, p2.name) && first_line == p2.first_line && first_column == p2.first_column &&
            last_line == p2.last_line && last_column == p2.last_column);
}

SourcePos Union(const SourcePos &p1, const SourcePos &p2) {
    if (strcmp(p1.name, p2.name) != 0)
        return p1;

    SourcePos ret;
    ret.name = p1.name;
    ret.first_line = std::min(p1.first_line, p2.first_line);
    ret.first_column = std::min(p1.first_column, p2.first_column);
    ret.last_line = std::max(p1.last_line, p2.last_line);
    ret.last_column = std::max(p1.last_column, p2.last_column);
    return ret;
}

} // namespace ispc
