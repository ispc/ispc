#!/bin/bash

# Copyright (c) 2023, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

set -e

function check_sha() {
    PRESET_FILE=$1
    REPO=$2

    CURRENT_SHA=$(jq -r '.configurePresets[0].cacheVariables.SPIRV_TRANSLATOR_SHA' ./ispc/superbuild/$PRESET_FILE)
    SPIRV_BRANCH=$(jq -r '.configurePresets[0].cacheVariables.SPIRV_TRANSLATOR_BRANCH' ./ispc/superbuild/$PRESET_FILE)
    TOP_SHA=$(cd $2; git checkout --quiet $SPIRV_BRANCH; git log --pretty -1 --format=format:%H)
    SHA_BRANCH_NAME="$SHA_BRANCH_NAME-${TOP_SHA: -8}"
    if [ "$CURRENT_SHA" == "$TOP_SHA" ]; then
        echo "Commit sha is the same $CURRENT_SHA in $PRESET_FILE - skipping..." >> $GITHUB_STEP_SUMMARY
        return
    fi

    echo "Updating SPIRV-LLVM-Translator sha in $PRESET_FILE from $CURRENT_SHA to $TOP_SHA" >> $GITHUB_STEP_SUMMARY

    jq -r ".configurePresets[0].cacheVariables.SPIRV_TRANSLATOR_SHA = \"$TOP_SHA\""  ./ispc/superbuild/$PRESET_FILE > new_preset_file
    mv new_preset_file ./ispc/superbuild/$PRESET_FILE
    any_updated=true
}

SCRIPTS_DIR=$( dirname -- "$( readlink -f -- "$0" )" )
source $SCRIPTS_DIR/update-common.sh

SHA_BRANCH_NAME=
any_updated=false
# Try to update version in all preset files
check_sha intPresets.json SPIRV-LLVM-Translator-int
check_sha osPresets.json SPIRV-LLVM-Translator
check_sha os-intPresets.json SPIRV-LLVM-Translator

if [ "$any_updated" = true ]; then
    BRANCH_NAME=robotex/update-spirv-translator$SHA_BRANCH_NAME
    COMMIT_MESSAGE="SPIRV-LLVM-Translator: update"

    cd ispc
    create_pr $BRANCH_NAME "$COMMIT_MESSAGE"
fi
