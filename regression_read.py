#!/usr/bin/python
#
#  Copyright (c) 2014, Intel Corporation 
#  All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
# 
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
# 
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
# 
#    * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
# 
# 
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
#   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# // Authors: Anton Mitrokhin

from optparse import OptionParser
from regression import *
import pickle



def read_test_table(filename):
    with open(filename, 'r') as fp:
        tt = pickle.load(fp) 
    return tt
        
        
if __name__ == '__main__':
    # parsing options
    class MyParser(OptionParser):
        def format_epilog(self, formatter):
            return self.epilog

    examples =  ("Examples:\n" +
    "Load test_table object\n\tregression_read.py -l\n")
    
    parser = MyParser(usage="Usage: regression_read.py -l [options]", epilog=examples)
    parser.add_option('-l', '--load-tt', dest='load_tt',
        help='load test_table.dump from file', default=None)

    (options, args) = parser.parse_args()

    tt = read_test_table(options.load_tt)
    print tt
    print "\n\n ------------------------\n\n"
    print "Avaluable LLVM revisions:", tt.table.keys()
    #print tt.regression(tt.table.keys()[1], tt.table.keys()[0])



