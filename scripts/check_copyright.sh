#!/bin/bash
# Copyright (c) 2022-2024, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

REF="$1"

export YEAR=$(date +%Y)

REPLARG="-S 1024"
if !(xargs $REPLARG 2>/dev/null); then
    # xargs doesn't support -S argument, so do not use it
    REPLARG=""
fi

# List files with Copyright whether they have up-to-date date.
git diff --name-only "$REF" | xargs -I {} $REPLARG bash -c 'test -f {} && (grep -q "Copyright.*$YEAR" {} || (grep -q Copyright {} && echo {}))' | ( ! grep ^ )
