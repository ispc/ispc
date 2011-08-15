#!/bin/bash

surprises=0
verbose=false
number=$(ls -1 tests/*.ispc|wc -l)
counter=1
target=sse4

while getopts ":vt:h" opt;do
    case $opt in
        v) verbose=true
            ;;
        t) target=$OPTARG
            ;;
        h) cat <<EOF
           usage: run_tests.sh [-v] [-t target] [filenames]
                  -v           # verbose output
                  -t           # specify compilation target (SSE4 is the default).
                  [filenames]  # (optional) files to run through testing infrastructure
                               # if none are provided, all in tests/ will be run.
EOF
            exit 1
    esac
done

ISPC_ARCH=x86-64
if [[ $OS == "Windows_NT" ]]; then
  ISPC_ARCH=x86
fi
ISPC_ARGS="--target=$target --arch=$ISPC_ARCH -O2 --woff"

shift $(( $OPTIND - 1 ))
if [[ "$1" > 0 ]]; then
    while [[ "$1" > 0 ]]; do
        i=$1
        shift
        echo Running test $i

        bc=${i%%ispc}bc
        ispc $ISPC_ARGS $i -o $bc --emit-llvm
        if [[ $? != 0 ]]; then
            surprises=1
            echo Test $i FAILED ispc compile
            echo
        else
            ispc_test $bc
            if [[ $? != 0 ]]; then
                surprises=1
                echo Test $i FAILED ispc_test
                echo
            fi
        fi
        /bin/rm -f $bc
    done
else
    echo Running all correctness tests

    for i in tests/*.ispc; do
        if $verbose; then
            echo -en "Running test $counter of $number.\r"
        fi
        (( counter++ ))
        bc=${i%%ispc}bc
        ispc $ISPC_ARGS $i -o $bc --emit-llvm 
        if [[ $? != 0 ]]; then
            surprises=1
            echo Test $i FAILED ispc compile
            echo
        else
            ispc_test $bc
            if [[ $? != 0 ]]; then
                surprises=1
                echo Test $i FAILED ispc_test
                echo
            fi
        fi
        /bin/rm -f $bc
    done

    echo -e "\nRunning failing tests"
    for i in failing_tests/*.ispc; do
        (ispc -O2 $i -woff -o - --emit-llvm | ispc_test -) 2>/dev/null 1>/dev/null
        if [[ $? == 0 ]]; then
            surprises=1
            echo Test $i UNEXPECTEDLY PASSED
            echo
        fi
    done
fi

if [[ $surprises == 0 ]]; then
    echo No surprises.
fi

exit $surprises
