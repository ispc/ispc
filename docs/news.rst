=========
ispc News
=========

ispc 1.1.4 is Released
----------------------

On February 4, 2012, the 1.1.4 release of ``ispc`` was posted; new features
include ``new`` and ``delete`` for dynamic memory allocation in ``ispc``
programs, "local" atomic operations in the standard library, and a new
scalar compilation target.  See the `1.1.4 release notes`_ for details.

.. _1.1.4 release notes: https://github.com/ispc/ispc/tree/master/docs/ReleaseNotes.txt


ispc 1.1.3 is Released
----------------------

With this release, the language now supports "switch" statements, with the same semantics and syntax as in C.

This release includes fixes for two important performance related issues:
the quality of code generated for "foreach" statements has been
substantially improved, and performance regression with code for "gathers"
that was introduced in v1.1.2 has been fixed in this release.

Thanks to Jean-Luc Duprat for a number of patches that improve support for
building on various platforms, and to Pierre-Antoine Lacaze for patches so
that ispc builds under MinGW.
