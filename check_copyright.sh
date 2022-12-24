#!/bin/bash
# Copyright 2022, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

REF="$1"

export YEAR=$(date +%Y)

# List files with Copyright whether they have up-to-date date.
git diff --name-only "$REF" | xargs -I {} bash -c 'test -f {} && (grep -q "Copyright.*$YEAR" {} || (grep -q Copyright {} && echo {}))' | ( ! grep ^ )
