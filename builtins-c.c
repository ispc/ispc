/*
  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  
*/

/** @file builtins-c.c
    @brief Standard library function implementations written in C.

    This file provides C implementations of various functions that can be
    called from ispc programs; in other words, this file is *not* linked
    into the ispc compiler executable, but rather provides functions that
    can be compiled into ispc programs.

    When the ispc compiler is built, this file is compiled with clang to
    generate LLVM bitcode.  This bitcode is later linked in to the program
    being compiled by the DefineStdlib() function.  The first way to access
    definitions from this file is by asking for them name from the
    llvm::Module's' symbol table (e.g. as the PrintStmt implementation does
    with __do_print() below.  Alternatively, if a function defined in this
    file has a signature that can be mapped back to ispc types by the
    lLLVMTypeToIspcType() function, then its declaration will be made
    available to ispc programs at compile time automatically.
  */


#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>

typedef int Bool;

#define PRINT_SCALAR(fmt, type)  \
    printf(fmt, *((type *)ptr)); \
    break

#define PRINT_VECTOR(fmt, type)                                         \
    putchar('[');                                                       \
    for (int i = 0; i < width; ++i) {                                   \
        /* only print the value if the current lane is executing */     \
        if (mask & (1<<i))                                              \
            printf(fmt, ((type *)ptr)[i]);                              \
        else                                                            \
            printf("((" fmt "))", ((type *)ptr)[i]);                    \
        putchar(i != width-1 ? ',' : ']');                              \
    }                                                                   \
    break

/** This function is called by PrintStmt to do the work of printing values
    from ispc programs.  Note that the function signature here must match
    the parameters that PrintStmt::EmitCode() generates.

    @param format  Print format string
    @param types   Encoded types of the values being printed.
                   (See lEncodeType()). 
    @param width   Vector width of the compilation target
    @param mask    Current lane mask when the print statemnt is called
    @param args    Array of pointers to the values to be printed
 */
void __do_print(const char *format, const char *types, int width, int mask, 
                void **args) {
    if (mask == 0) 
        return;

    int argCount = 0;
    while (*format) {
        // Format strings are just single percent signs.
        if (*format != '%')
            putchar(*format);
        else {
            if (*types) {
                void *ptr = args[argCount++];
                // Based on the encoding in the types string, cast the
                // value appropriately and print it with a reasonable
                // printf() formatting string.
                switch (*types) {
                case 'b': {
                    printf("%s", *((Bool *)ptr) ? "true" : "false");
                    break;
                }
                case 'B': {
                    putchar('[');
                    for (int i = 0; i < width; ++i) {
                        if (mask & (1<<i))
                            printf("%s", ((Bool *)ptr)[i] ? "true" : "false");
                        else
                            printf("_________");
                        putchar(i != width-1 ? ',' : ']');
                    }
                    break;
                }
                case 'i': PRINT_SCALAR("%d", int);
                case 'I': PRINT_VECTOR("%d", int);
                case 'u': PRINT_SCALAR("%u", unsigned int);
                case 'U': PRINT_VECTOR("%u", unsigned int);
                case 'f': PRINT_SCALAR("%f", float);
                case 'F': PRINT_VECTOR("%f", float);
                case 'l': PRINT_SCALAR("%lld", long long);
                case 'L': PRINT_VECTOR("%lld", long long);
                case 'v': PRINT_SCALAR("%llu", unsigned long long);
                case 'V': PRINT_VECTOR("%llu", unsigned long long);
                case 'd': PRINT_SCALAR("%f", double);
                case 'D': PRINT_VECTOR("%f", double);
                default:
                    printf("UNKNOWN TYPE ");
                    putchar(*types);
                }
                ++types;
            }
        }
        ++format;
    }
    fflush(stdout);
}
