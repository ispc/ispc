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

# // Authors: Anton Mitrokhin, Vsevolod Livinskiy

class TestResult(object):
    """
    this class stores basicly two integers which stand for the result
    of the test: (runfail{0/1}, compfail{0/1}). other values are 
    deemed invalid. the __cmp__ function of this class is used to 
    define what test regression actually is.
    """
    def __init__(self, runfailed, compfailed):
        self.runfailed, self.compfailed = (runfailed, compfailed)

    def __cmp__(self, other):
        if isinstance(other, TestResult):
            if self.runfailed == other.runfailed   and \
               self.compfailed == other.compfailed:
                return 0
            elif self.compfailed > other.compfailed:
                return 1
            elif self.runfailed > other.runfailed and \
                 self.compfailed == other.compfailed:
                return 1
            else:
                return -1

        raise RuntimeError("Wrong type for comparioson")
        return NotImplemented

    def __repr__(self):
        if (self.runfailed < 0 or self.compfailed < 0):
            return "(Undefined)"
        return "(r%d c%d)" % (self.runfailed, self.compfailed)


class TestCase(object):
    """
    the TestCase() is a combination of parameters the tast was run with:
    the architecture (x86, x86-64 ...), compiler optimization (-O0, -O2 ...)
    and target (sse, avx ...). we also store the result of the test here.
    """
    def __init__(self, arch, opt, target):
        self.arch, self.opt, self.target = (arch, opt, target)
        self.result = TestResult(-1, -1)

    def __repr__(self):
        string = "%s %s %s: " % (self.arch, self.opt, self.target)
        string = string + repr(self.result)
        return string

    def __hash__(self):
        return hash(self.arch + self.opt + self.target)

    def __ne__(self, other):
        if isinstance(other, TestCase):
            if hash(self.arch + self.opt + self.target) != hash(other):
                return True
            return False
        raise RuntimeError("Wrong type for comparioson")
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, TestCase):
            return not self.__ne__(other)   
        raise RuntimeError("Wrong type for comparioson")
        return NotImplemented


class Test(object):
    """
    Test() stores all TestCase() objects for a given test file name
    i.e. all archs/opts/targets/ and corresponding testing results.
    """
    def __init__(self, name):
        self.name = name
        self.test_cases = []

    def add_result(self, test_case):
        if test_case in self.test_cases:
            raise RuntimeError("This test case is already in the list: " + repr(test_case))
            return
        self.test_cases.append(test_case)

    def __repr__(self):
        string = self.name + '\n'
        string = string.rjust(20)
        for test_case in self.test_cases:
            string += repr(test_case).rjust(60) + '\n'
        return string
    
    def __hash__(self):
        return hash(self.name)

    def __ne__(self, other):
        if isinstance(other, Test):
            if hash(self) != hash(other):
                return True
            return False
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Test):
            return not self.__ne__(other)   
        return NotImplemented


class RegressionInfo(object):
    """
    service class which provides some statistics on a given regression.
    the regression test names and cases are given in a form of Test() objects
    with empty (-1, -1) results 
    """
    def __init__(self, revision_old, revision_new, tests):
        self.revision_old, self.revision_new = (revision_old, revision_new)
        self.tests = tests
        self.archfailes = {}
        self.optfails = {}
        self.targetfails = {}
        self.testfails = {}
        self.archs = []
        self.opts = []
        self.targets = []

        for test in tests:
            for test_case in test.test_cases:
                self.inc_dictionary(self.testfails, test.name)
                self.inc_dictionary(self.archfailes, test_case.arch)
                self.inc_dictionary(self.optfails, test_case.opt)
                self.inc_dictionary(self.targetfails, test_case.target)
        
        self.archs = self.archfailes.keys()
        self.opts = self.optfails.keys()
        self.targets = self.targetfails.keys()

    def inc_dictionary(self, dictionary, key):
        if key not in dictionary:
            dictionary[key] = 0
        dictionary[key] += 1

    def __repr__(self):
        string = "Regression of LLVM revision %s in comparison to %s\n" % (self.revision_new, self.revision_old)
        string += repr(self.tests) + '\n'
        string += str(self.testfails) + '\n'
        string += str(self.archfailes) + '\n'
        string += str(self.optfails) + '\n'
        string += str(self.targetfails) + '\n'

        return string

        
class TestTable(object):
    """
    the table which stores a tuple of Test() objects (one per revision) and has some 
    convenience methods for dealing with them
    """
    def __init__(self):
        """ This dictionary contains {rev: [test1, test2, ...], ...}, where 'rev' is a string (revision name) and 'test#'
        is a Test() object instance """
        self.table = {}

    def add_result(self, revision_name, test_name, arch, opt, target, runfailed, compfailed):
        revision_name = str(revision_name)
        if revision_name not in self.table:
            self.table[revision_name] = []
        
        test_case = TestCase(arch, opt, target)
        test_case.result = TestResult(runfailed, compfailed)

        for test in self.table[revision_name]:
            if test.name == test_name:
                test.add_result(test_case)
                return
        
        test = Test(test_name)
        test.add_result(test_case)
        self.table[revision_name].append(test)


    def test_intersection(self, test1, test2):
        """ Return test cases common for test1 and test2. If test names are different than there is nothing in common """
        if test1.name != test2.name:
            return []
        return list(set(test1.test_cases) & set(test2.test_cases))

    def test_regression(self, test1, test2):
        """ Return the tuple of empty (i.e. with undefined results) TestCase() objects 
            corresponding to regression in test2 comparing to test1 """
        if test1.name != test2.name:
            return []

        regressed = []
        for tc1 in test1.test_cases:
            for tc2 in test2.test_cases:
                """ If test cases are equal (same arch, opt and target) but tc2 has more runfails or compfails """
                if tc1 == tc2 and tc1.result < tc2.result:
                    regressed.append(TestCase(tc1.arch, tc1.opt, tc1.target))
        return regressed

    def regression(self, revision_old, revision_new):
        """ Return a tuple of Test() objects containing TestCase() object which show regression along given revisions """
        revision_old, revision_new = (str(revision_old), str(revision_new))
        if revision_new not in self.table:
            raise RuntimeError("This revision in not in the database: " + str(revision_new) + " (" + str(self.table.keys()) + ")")
            return

        if revision_old not in self.table:
            raise RuntimeError("This revision in not in the database: " + str(revision_old) + " (" + str(self.table.keys()) + ")")
            return

        regressed = []
        for test_old in self.table[revision_old]:
            for test_new in self.table[revision_new]:
                tr = self.test_regression(test_old, test_new)
                if len(tr) == 0:
                    continue
                test = Test(test_new.name)
                for test_case in tr:
                    test.add_result(test_case)
                regressed.append(test)
        return RegressionInfo(revision_old, revision_new, regressed)
    
    def __repr__(self):
        string = ""
        for rev in self.table.keys():
            string += "[" + rev + "]:\n"
            for test in self.table[rev]:
                string += repr(test) + '\n'
        return string

 

class RevisionInfo(object):
    """
    this class is intended to store some relevant information about curent LLVM revision
    """
    def __init__(self, hostname, revision):
        self.hostname, self.revision = hostname, revision
        self.archs = []
        self.opts = []
        self.targets = []
        self.succeed = 0
        self.runfailed = 0
        self.compfailed = 0
        self.skipped = 0
        self.testall = 0
        self.regressions = {}
    
    def register_test(self, arch, opt, target, succeed, runfailed, compfailed, skipped):
        if arch not in self.archs:
            self.archs.append(arch)
        if opt not in self.opts:
            self.opts.append(opt)
        if target not in self.targets:
            self.targets.append(target)
        self.runfailed += runfailed
        self.compfailed += compfailed
        self.skipped += skipped
        self.succeed += succeed

    def add_regression(self, revision, regression_info):
        """ input is intended to be from 'TestTable.regression(..)', 'regression_info' is a tuple of RegressionInfo() object
        (regression.py) and 'revision' is tested (not current) LLVM revision name """
        if revision == self.revision:
            raise RuntimeError("No regression can be found along the same LLVM revision!")
      
        if revision in self.regressions:
            raise RuntimeError("This revision regression info is already in self.regressions!")
      
        self.regressions[revision] = regression_info

    def __repr__(self):
        string = "%s: LLVM(%s)\n" % (self.hostname, self.revision)
        string += "archs  : %s\n" % (str(self.archs))
        string += "opts   : %s\n" % (str(self.opts))
        string += "targets: %s\n" % (str(self.targets))
        string += "runfails: %d/%d\n" % (self.runfailed, self.testall)
        string += "compfails: %d/%d\n" % (self.compfailed, self.testall)
        string += "skipped: %d/%d\n" % (self.skipped, self.testall)
        string += "succeed: %d/%d\n" % (self.succeed, self.testall)
        return string




class ExecutionStateGatherer(object):
    def __init__(self):
        self.hostname = self.get_host_name()
        self.revision = ""
        self.rinf = []
        self.tt = TestTable()

    def switch_revision(self, revision):
        print "Switching revision to " + revision
        self.revision = revision
        self.rinf.append(RevisionInfo(self.hostname, self.revision))

    def current_rinf(self):
        if len(self.rinf) == 0:
            raise RuntimeError("self.rinf is empty. Apparently you've never invoked switch_revision")
        return self.rinf[len(self.rinf) - 1]

    def add_to_tt(self, test_name, arch, opt, target, runfailed, compfailed):
        if len(self.rinf) == 0:
            raise RuntimeError("self.rinf is empty. Apparently you've never invoked switch_revision")
        self.tt.add_result(self.revision, test_name, arch, opt, target, runfailed, compfailed)

    def add_to_rinf(self, arch, opt, target, succeed, runfailed, compfailed, skipped):
        self.current_rinf().register_test(arch, opt, target, succeed, runfailed, compfailed, skipped)

    def add_to_rinf_testall(self, tried_to_test):
        self.current_rinf().testall += tried_to_test

    def load_from_tt(self, tt):
        # TODO: fill in self.rinf field!
        self.tt = tt
        REVISIONS = tt.table.keys()
        self.revision = ""
        if len(REVISIONS) != 0:
            self.revision = REVISIONS[0]
        print "ESG: loaded from 'TestTable()' with revisions", REVISIONS

    def dump(self, fname, obj):
        import pickle
        with open(fname, 'w') as fp:
            pickle.dump(obj, fp)  

    def undump(self, fname):
        import pickle
        with open(fname, 'r') as fp:
            obj = pickle.load(fp) 
        return obj

    def get_host_name(self):
        import socket
        return socket.gethostname()

    def __repr__(self):
        string = "Hostname: %s\n" % (self.hostname)
        string += "Current LLVM Revision = %s\n\n" % (self.revision)
        for rev_info in self.rinf:
            string += repr(rev_info) + '\n'
        return string


