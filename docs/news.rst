=========
ispc News
=========

ispc 1.2.1 is Released
----------------------

This is a bugfix release, fixing approximately 20 bugs in the system and
improving error handling and error reporting.  New functionality includes
very efficient float/half conversion routines thanks to Fabian 
Giesen.  See the `1.2.1 release notes`_ for details.

.. _1.2.1 release notes: https://github.com/ispc/ispc/tree/master/docs/ReleaseNotes.txt

ispc 1.2.0 is Released
-----------------------

A new major release was posted on March 20, 2012.  This release includes
significant new functionality for cleanly handling "structure of arrays"
(SoA) data layout and a new model for how uniform and varying are handled
with structure types.  

Paper on ispc To Appear in InPar 2012
-------------------------------------

A technical paper on ``ispc``, `ispc: A SPMD Compiler for High-Performance
CPU Programming`_, by Matt Pharr and William R. Mark, has been accepted to
the `InPar 2012`_ conference. This paper describes a number of the design
features and key characteristics of the ``ispc`` implementation.

(© 2012 IEEE. Personal use of this material is permitted. Permission from
IEEE must be obtained for all other uses, in any current or future media,
including reprinting/republishing this material for advertising or
promotional purposes, creating new collective works, for resale or
redistribution to servers or lists, or reuse of any copyrighted component
of this work in other works.).

.. _ispc\: A SPMD Compiler for High-Performance CPU Programming: https://github.com/downloads/ispc/ispc/ispc_inpar_2012.pdf
.. _InPar 2012: http://innovativeparallel.org/

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
