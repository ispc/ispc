#ifndef ISPC_OPTIONS_H
#define ISPC_OPTIONS_H

void PrintVersion();

void PrintUsage(int ret);

void PrintDevUsage(int ret);

enum OptionParseResult {
    OPTION_PARSE_RESULT_SUCCESS = 0,

    OPTION_PARSE_RESULT_ERROR = 1,

    OPTION_PARSE_RESULT_REQUEST_USAGE = 2,
    OPTION_PARSE_RESULT_REQUEST_DEV_USAGE = 4,

    OPTION_PARSE_RESULT_REQUEST_VERSION = 8
};

inline OptionParseResult operator|(OptionParseResult a, OptionParseResult b) {
    return static_cast<OptionParseResult>(
        static_cast<int>(a) | static_cast<int>(b));
}

OptionParseResult ParseOptions(int Argc, char* Argv[]);

#endif

