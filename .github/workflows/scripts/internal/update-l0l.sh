#!/bin/bash

# Copyright (c) 2023, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

set -e

function check_sha() {
    PRESET_FILE=$1

    CURRENT_TAG=$(jq -r '.configurePresets[0].cacheVariables.L0_TAG' ./ispc/superbuild/$PRESET_FILE)
    if [ "$CURRENT_TAG" == "$TOP_TAG" ]; then
        echo "L0 tag is the same $CURRENT_TAG in $PRESET_FILE - skipping..." >> $GITHUB_STEP_SUMMARY
        return
    fi

    echo "Updating Level-Zero version in $PRESET_FILE from $CURRENT_TAG to $TOP_TAG" >> $GITHUB_STEP_SUMMARY

    jq -r ".configurePresets[0].cacheVariables.L0_TAG = \"$TOP_TAG\""  ./ispc/superbuild/$PRESET_FILE > new_preset_file
    mv new_preset_file ./ispc/superbuild/$PRESET_FILE
    any_updated=true
}

SCRIPTS_DIR=$( dirname -- "$( readlink -f -- "$0" )" )
source $SCRIPTS_DIR/update-common.sh

TOP_TAG=$(cd level-zero; git tag --sort=committerdate | tail -n1)

any_updated=false
# Try to update version in all preset files
check_sha intPresets.json
check_sha osPresets.json
check_sha os-intPresets.json

if [ "$any_updated" = true ]; then
    BRANCH_NAME=robotex/update-l0l-$TOP_TAG
    COMMIT_MESSAGE="level-zero: update to $TOP_TAG"

    cd ispc
    create_pr $BRANCH_NAME "$COMMIT_MESSAGE"
fi
