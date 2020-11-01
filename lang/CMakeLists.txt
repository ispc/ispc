cmake_minimum_required(VERSION 3.9.6)

project(ispc-lang)

option(ISPC_LANG_TESTING "Whether or not to enable testing (requires Googletest)." OFF)

set(gendir "${CMAKE_CURRENT_BINARY_DIR}/generated_sources")

if(NOT EXISTS "${gendir}")
    file(MAKE_DIRECTORY "${gendir}")
endif(NOT EXISTS "${gendir}")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(cxxflags -Wall -Wextra -Werror -Wfatal-errors)
endif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(cxxflags -Wall -Wextra -Wdocumentation -Werror -Wfatal-errors)
endif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

###############
# Bison Setup #
###############

find_package(BISON 3.0 REQUIRED)

set(BISON_INPUT src/parser.yy)

set(BISON_CPP_OUTPUT ${gendir}/parser.cc)

set(BISON_OUTPUT ${gendir}/parser.hh
                 ${gendir}/parser.output)

add_custom_command(
    OUTPUT ${BISON_CPP_OUTPUT} ${BISON_OUTPUT}
    COMMAND ${BISON_EXECUTABLE} -d -t -v
            --output=${BISON_CPP_OUTPUT}
            ${BISON_INPUT}
    DEPENDS ${BISON_INPUT}
    COMMENT "Generating parse.cc"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

##############
# Flex Setup #
##############

find_package(FLEX 2.5 REQUIRED)

set(FLEX_INPUT  src/scanner.ll)

set(FLEX_OUTPUT ${gendir}/scanner.cpp)
set(FLEX_HEADER ${gendir}/scanner.h)

add_custom_command(
    OUTPUT ${FLEX_OUTPUT} ${FLEX_HEADER}
    COMMAND ${FLEX_EXECUTABLE}
            --outfile=${FLEX_OUTPUT}
            --header-file=${FLEX_HEADER}
            ${FLEX_INPUT}
    DEPENDS ${FLEX_INPUT}
    COMMENT "Generating lex.cpp"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

###########
# Library #
###########

# Generated sources go into separate
# target so that they can opt out of
# receiving compiler warnings.
add_library(ispc-lang-generated OBJECT
    ${FLEX_OUTPUT}
    ${FLEX_HEADER}
    ${BISON_CPP_OUTPUT})

target_include_directories(ispc-lang-generated PRIVATE include)

target_compile_features(ispc-lang-generated PRIVATE cxx_std_17)

add_library(ispc-lang
    include/ispc/ast_node.h
    include/ispc/ast_node_consumer.h
    include/ispc/ast_node_visitor.h
    include/ispc/diagnostic.h
    include/ispc/diagnostic_consumer.h
    include/ispc/diagnostic_printer.h
    include/ispc/expr.h
    include/ispc/parser.h
    include/ispc/scanner.h
    include/ispc/token.h
    include/ispc/token_consumer.h
    src/expr.cpp
    src/diagnostic_printer.cpp
    src/parser.cpp
    src/scanner.cpp
    src/invalid_token_diagnostic.h
    src/invalid_token_diagnostic.cpp
    $<TARGET_OBJECTS:ispc-lang-generated>)

target_compile_options(ispc-lang PRIVATE ${cxxflags})

target_include_directories(ispc-lang
    PUBLIC include
    PRIVATE "${gendir}")

target_compile_features(ispc-lang PRIVATE cxx_std_17)

add_library(ISPC::Lang ALIAS ispc-lang)

#########
# Tests #
#########

if(ISPC_LANG_TESTING)

    find_package(GTest REQUIRED)

    add_executable(ispc-lang-tests
        testing/main.cpp
        testing/diagnostic_test.h
        testing/diagnostic_test.cpp
        testing/parser_tests.cpp
        testing/scanner_tests.cpp)

    set_target_properties(ispc-lang-tests
        PROPERTIES
            OUTPUT_NAME run_tests)

    target_compile_definitions(ispc-lang-tests
        PRIVATE TESTING_DIR="\"\"${CMAKE_CURRENT_SOURCE_DIR}/testing\"\"")

    target_compile_features(ispc-lang-tests
        PRIVATE cxx_std_17)

    target_link_libraries(ispc-lang-tests
        PRIVATE ISPC::Lang GTest::GTest GTest::Main)

    add_test(NAME ispc-lang-tests
        COMMAND $<TARGET_FILE:ispc-lang-tests>
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/testing")

    enable_testing()

endif(ISPC_LANG_TESTING)
