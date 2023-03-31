/*
  Copyright (c) 2013, Durham University
  Copyright (c) 2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/* Author: Tomasz Koziara */

#include <algorithm>
#include <utility>
#include <vector>

typedef std::pair<double, int> pair;

struct cmp {
    bool operator()(const pair &a, const pair &b) { return a.first < b.first; }
};

void sort_serial(int n, unsigned int code[], int order[]) {
    std::vector<pair> pairs;

    pairs.reserve(n);

    for (int i = 0; i < n; i++)
        pairs.push_back(pair(code[i], i));

    std::sort(pairs.begin(), pairs.end(), cmp());

    int *o = order;

    for (std::vector<pair>::const_iterator p = pairs.begin(); p != pairs.end(); ++p, ++o)
        *o = p->second;
}
