#!/bin/bash

# Copyright (c) 2023, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

function create_pr() {
    BRANCH_NAME="$1"
    COMMIT_MESSAGE="$2"

    if [ -z "$BRANCH_NAME" ]; then
        echo "Empty BRANCH_NAME. Aborting."
        exit 1
    fi

    git config --global user.email "arina.neshlyaeva@intel.com"
    git config --global user.name "RobotEx"
    git diff
    git checkout -b $BRANCH_NAME
    # It is safe git add non-changed file.
    git add ./superbuild/osPresets.json
    git commit -m "$COMMIT_MESSAGE" || true
    git add ./superbuild/intPresets.json
    git add ./superbuild/os-intPresets.json
    git commit -m "CI (internal): $COMMIT_MESSAGE" || true
    git push origin $BRANCH_NAME

    # create PR
    JSON='{"title":"'$COMMIT_MESSAGE'","head":"'$BRANCH_NAME'","base":"gen"}'
    RESPOND=$(curl -X POST -H "Accept: application/vnd.github+json" -H "Authorization: token $ACCESS_TOKEN" https://api.github.com/repos/intel-innersource/applications.compilers.ispc.core/pulls -d "$JSON")
    PR_ID=$(echo $RESPOND | jq -r '.number')

    # Add RobotEx label to PR
    curl -X PUT -H "Accept: application/vnd.github+json" -H "Authorization: token $ACCESS_TOKEN" https://api.github.com/repos/intel-innersource/applications.compilers.ispc.core/issues/$PR_ID/labels -d '{"labels":["RobotEx"]}' >> $GITHUB_STEP_SUMMARY

    echo "### PR" >> $GITHUB_STEP_SUMMARY
    echo "$(echo $RESPOND | jq -r '.html_url')" >> $GITHUB_STEP_SUMMARY
}
