#!/bin/bash

# Copyright 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# Author: Giordano Salvador <giordano.salvador@intel.com>
#
# Description: A simple whole word search for unsafe function names.
# May produce false positives if the names appear in comments or other text.


VERBOSE=OFF
EXT_DEFAULT="m4 h hpp c cpp"
EXTS=${EXTS=$EXT_DEFAULT}


UNSAFE_SET=(
    UNSAFE_I327
    UNSAFE_I328
    UNSAFE_I329
    UNSAFE_I330
    UNSAFE_I331
    UNSAFE_I332
    UNSAFE_I333
    UNSAFE_I334
    UNSAFE_I335
    UNSAFE_I336
    UNSAFE_I337
    UNSAFE_I340
    UNSAFE_I341
    UNSAFE_I342
    UNSAFE_I343
    UNSAFE_I344
    UNSAFE_I345
    UNSAFE_I347
) # End UNSAFE_SET

UNSAFE_I327=(
    StrCat
    StrCatA
    StrCatBuff
    StrCatBuffA
    StrCatBuffW
    StrCatChainW
    StrCatN
    StrCatNA
    StrCatNW
    StrCatW
    StrNCat
    StrNCatA
    StrNCatW
    _ftcscat
    _mbccat
    _mbscat
    _mbsnbcat
    _mbsncat
    _tccat
    _tcscat
    _tcsncat
    lstrcat
    lstrcatA
    lstrcatW
    lstrcatn
    lstrcatnA
    lstrcatnW
    lstrncat
    strcatA
    strcatW
    strncat
    wcscat
    wcsncat
    strcat
) # End UNSAFE_I327

UNSAFE_I328=(
    strcpy
    wcsncpy
    wcscpy
    strncpy
    strcpynA
    strcpyW
    strcpyA
    lstrcpynW
    lstrcpynA
    lstrcpyn
    lstrcpyW
    lstrcpyA
    lstrcpy
    _tcsncpy
    _tcscpy
    _tccpy
    _mbsncpy
    _mbsnbcpy
    _mbscpy
    _mbccpy
    _ftcscpy
    StrNCpyW
    StrNCpyA
    StrNCpy
    StrCpyW
    StrCpyNW
    StrCpyNA
    StrCpyN
    StrCpyA
    StrCpy
) # End UNSAFE_I328

UNSAFE_I329=(
    sprintfW
    wvsprintfW
    wvsprintfA
    wvsprintf
    wsprintfW
    wsprintfA
    wsprintf
    vswprintf
    vsprintf
    swprintf
    sprintfA
    sprintf
    _vstprintf
    _stprintf
) # End UNSAFE_I329

UNSAFE_I330=(
    wvsprintf
    wvsprintfW
    wvsprintfA
    vswprintf
    vsprintf
    _vstprintf
) # End UNSAFE_I330

UNSAFE_I331=(
    strncpy
    wcsncpy
    strcpynA
    lstrcpynW
    lstrcpynA
    lstrcpyn
    _tcsncpy
    _mbsncpy
    _mbsnbcpy
    _fstrncpy
    StrNCpyW
    StrNCpyA
    StrNCpy
    StrCpyNW
    StrCpyNA
    StrCpyN
) # End UNSAFE_I331

UNSAFE_I332=(
    strncat
    wcsncat
    lstrncat
    lstrcatnW
    lstrcatnA
    lstrcatn
    _tcsncat
    _mbsncat
    _mbsnbcat
    _fstrncat
    StrNCatW
    StrNCatA
    StrNCat
    StrCatNW
    StrCatNA
    StrCatN
) # End UNSAFE_I332

UNSAFE_I333=(
    gets
    _getts
    _gettws
) # End UNSAFE_I333

UNSAFE_I334=(
    memcpy
    wmemcpy
    RtlCopyMemory
    CopyMemory
) # End UNSAFE_I334

UNSAFE_I335=(
    IsBadCodePtr
    IsBadHugeReadPtr
    IsBadHugeWritePtr
    IsBadReadPtr
    IsBadStringPtr
    IsBadWritePtr
) # End UNSAFE_I335

UNSAFE_I336=(
    setjmp
    longjmp
) # End UNSAFE_I336

UNSAFE_I337=(
    istream
    cin
) # End UNSAFE_I337

UNSAFE_I340=(
    getwd
) # End UNSAFE_I340

UNSAFE_I341=(
    strlen
    wcslen
) # End UNSAFE_I341

UNSAFE_I342=(
    strlen
    wcslen
    _mbslen
    _mbstrlen
    StrLen
    lstrlen
) # End UNSAFE_I342

UNSAFE_I343=(
    gets
) # End UNSAFE_I343

UNSAFE_I344=(
    strcpy
    wcscpy
    strcat
    wcscat
) # End UNSAFE_I344

UNSAFE_I345=(
    sprintf
    vsprintf
) # End UNSAFE_I345

UNSAFE_I347=(
    asctime
) # End UNSAFE_I347


ret=0

function find-unsafe-fun
{
    local _set=$1
    local file=$2
    local fun=$3

    local options=(--with-filename --line-number --initial-tab --word-regexp)
    local output=$(grep ${options[@]} $fun $file)

    if [ -n "$output" ]; then
        echo ">> Found unsafe function '$fun' from set '$_set' in '$file'."
        case $VERBOSE in
          ON|on|TRUE|True|true) echo $output ;;
          *) ;;
        esac
        ret=1
    fi
}


msg_usage="$0 <source-path>"
msg_options="
    environment variables:

      EXT                             default: ${EXT_DEFAULT}

    command line :

      help|-help|--help|-h            Print this message.

      verbose|-verbose|--verbose|-v   Print more output.
" # End msg_options

if [ $# -lt 1 ]; then
    echo $msg_usage
    exit 1
fi

while [ -n "$1" ]; do
    case $1 in
        help|-help|--help|-h)
            echo $msg_usage
            echo $msg_options
            exit 0
            ;;
        verbose|-verbose|--verbose|-v)
            VERBOSE=ON
            ;;
        *)
            SRC_DIR=$1
            if [ ! -d $SRC_DIR ]; then
                echo ">> Failed to find source directory '$SRC_DIR'."
                exit 1
            fi
    esac
    shift
done

if [ -z $SRC_DIR ]; then
    SRC_DIR=$(pwd)
fi

for ext in ${EXTS}; do
    for file in $(find $SRC_DIR -name "*.$ext"); do
        for _set in ${UNSAFE_SET[@]}; do
            for fun in ${!_set}; do
                find-unsafe-fun $_set $file $fun
            done
        done
    done
done
exit $ret
