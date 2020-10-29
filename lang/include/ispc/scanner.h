#ifndef ISPC_LANG_INCLUDE_ISPC_SCANNER_H
#define ISPC_LANG_INCLUDE_ISPC_SCANNER_H

#include <memory>

#include <cstddef>

namespace ispc {

class ScannerImpl;
class TokenConsumer;

/** This is a wrapper around the
   generated Flex scanner. It is primarily
   meant to work with string buffers and is
   reentrant.
   */
class Scanner final {
    /** A pointer to the implementation data. */
    ScannerImpl *self = nullptr;
public:
    constexpr Scanner() noexcept {}

    constexpr Scanner(Scanner &other) noexcept : self(other.self) {
        other.self = nullptr;
    }

    ~Scanner();

    void AddTokenConsumer(std::unique_ptr<TokenConsumer> &&consumer);

    /** This function will scan for tokens
     * in @p text up until @p length.
     *
     * @note This function changes the state of the scanner.
     *
     * @param text The text to be scanned. This does
     *             not have to be null terminated.
     *
     * @param length The number of bytes in @p text.
     * */
    void ScanBuffer(const char *text, std::size_t length);

    /** Opens and scans a file at @p path.
     *
     * @param path The path of the file to open for scanning.
     *
     * @return True on success, false on failure.
     *         Check errno for a reason of failure.
     * */
    bool ScanFile(const char *path);
};

} // namespace ispc

#endif // ISPC_LANG_INCLUDE_ISPC_SCANNER_H
