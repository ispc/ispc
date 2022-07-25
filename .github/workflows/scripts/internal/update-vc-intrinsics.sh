#!/bin/bash

# Copyright (c) 2023, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

set -e

function check_sha() {
    PRESET_FILE=$1
    SHA=$2

    CURRENT_SHA=$(jq -r '.configurePresets[0].cacheVariables.VC_INTRINSICS_SHA' ./ispc/superbuild/$PRESET_FILE)
    if [ "$CURRENT_SHA" == "$SHA" ]; then
        echo "Commit sha is the same $CURRENT_SHA in $PRESET_FILE - skipping..." >> $GITHUB_STEP_SUMMARY
        return
    fi

    echo "Updating vc-intrinsics sha in $PRESET_FILE from $CURRENT_SHA to $SHA" >> $GITHUB_STEP_SUMMARY

    jq -r ".configurePresets[0].cacheVariables.VC_INTRINSICS_SHA = \"$SHA\""  ./ispc/superbuild/$PRESET_FILE > new_preset_file
    mv new_preset_file ./ispc/superbuild/$PRESET_FILE
    any_updated=true
}

SCRIPTS_DIR=$( dirname -- "$( readlink -f -- "$0" )" )
source $SCRIPTS_DIR/update-common.sh

# --quiet is needed to avoid extra output from checkout command
TOP_SHA_INT=$(cd vc-intrinsics-int; git checkout --quiet cmc_experimental; git log --pretty -1 --format=format:%H)
TOP_SHA=$(cd vc-intrinsics; git log --pretty -1 --format=format:%H)

any_updated=false
# Try to update version in all preset files
check_sha intPresets.json $TOP_SHA_INT
check_sha osPresets.json $TOP_SHA
check_sha os-intPresets.json $TOP_SHA

if [ "$any_updated" = true ]; then
    BRANCH_NAME=robotex/update-vc-intrinsics-${TOP_SHA_INT: -8}-${TOP_SHA: -8}-${TOP_SHA: -8}
    COMMIT_MESSAGE="vc-intrinsics: update"

    cd ispc
    create_pr $BRANCH_NAME "$COMMIT_MESSAGE"
fi
