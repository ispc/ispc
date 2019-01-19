=========
ispc News
=========

ispc 1.10.0 is Released
-----------------------

An ``ispc`` release with support for streaming stores/loads, 64 bit wide types
in aos_to_soa/soa_to_aos intrinsics, and a pragma to suppress warnings.

The release is based on a patched LLVM 5.0.2 backend.

For more details, please check `Release Notes`_.

.. _Release Notes: https://github.com/ispc/ispc/blob/master/docs/ReleaseNotes.txt

ispc 1.9.2 is Released
----------------------

An ``ispc`` release with out-of-the-box debug support on Windows and significant
performance improvements on AVX512 targets. Release is based on patched version
LLVM 5.0 backend.

For more details, please check `Release Notes`_.

.. _Release Notes: https://github.com/ispc/ispc/blob/master/docs/ReleaseNotes.txt

ispc 1.9.1 is Released
----------------------

An ``ispc`` release with new native AVX512 target for future Xeon CPUs and
improvements for debugging. Release is based on patched version LLVM 3.8 backend.

For more details, please check `Release Notes`_.

.. _Release Notes: https://github.com/ispc/ispc/blob/master/docs/ReleaseNotes.txt

ispc 1.9.0 is Released
----------------------

An ``ispc`` release with AVX512 (KNL flavor) support and a number of bug fixes,
based on fresh LLVM 3.8 backend.

For more details, please check `Release Notes`_.

.. _Release Notes: https://github.com/ispc/ispc/blob/master/docs/ReleaseNotes.txt

ispc 1.8.2 is Released
----------------------

An update of ``ispc`` with several important stability fixes and an experimental
AVX512 support has been released. Binaries are based on LLVM 3.6.1. Binaries with
native AVX512 support are based on LLVM 3.7 (r238198).

For more details, please check `Release Notes`_.

.. _Release Notes: https://github.com/ispc/ispc/blob/master/docs/ReleaseNotes.txt

ispc 1.8.1 is Released
----------------------

A minor update of ``ispc`` with several important stability fixes has been
released. Problem with auto-dispatch on Linux is fixed (affects only pre-built
binaries), the problem with -O2 -g is also fixed. There are several
improvements in Xeon Phi support. Similar to 1.8.0 all binaries are based on
LLVM 3.5.

ispc 1.8.0 is Released
----------------------

A major new version of ``ispc``, which introduces experimental support for NVPTX
target, brings numerous improvements to our KNC (Xeon Phi) support, introduces
debugging support on Windows and fixes several bugs. We also ship experimental
build for Sony PlayStation4 target in this release. Binaries for all platforms
are based on LLVM 3.5.

ispc 1.7.0 is Released
----------------------

A major new version of ``ispc`` with several language and library extensions and
fixes in debug info support. Binaries for all platforms are based on patched
version on LLVM 3.4. There also performance improvements beyond switchover to
LLVM 3.4.

ispc 1.6.0 is Released
----------------------

A major update of ``ispc`` has been released. The main focus is on improved 
performance and stability. Several new targets were added. There are also 
a number of language and library extensions. Released binaries are based on
patched LLVM 3.3 on Linux and MacOS and LLVM 3.4rc3 on Windows. Please refer
to Release Notes for complete set of changes.

ispc 1.5.0 is Released
----------------------

A major update of ``ispc`` has been released with several new targets available
and bunch of performance and stability fixes. The released binaries are built
with patched version of LLVM 3.3. Please refer to Release Notes for complete
set of changes.

ispc 1.4.4 is Released
----------------------

A minor update of ``ispc`` has been released with several stability improvements.
The released binaries are built with patched version of LLVM 3.3. Since this
release we also distribute 32 bit Linux binaries.

ispc 1.4.3 is Released
----------------------

A minor update of ``ispc`` has been released with several stability improvements.
All tests and examples now properly compile and execute on native targets on
Unix platforms (Linux and MacOS).
The released binaries are built with patched version of LLVM 3.3.

ispc 1.4.2 is Released
----------------------

A minor update of ``ispc`` has been released with stability fix for AVX2
(Haswell), fix for Win32 platform and performance improvements on Xeon Phi.
As usual, it's available on all supported platforms (Windows, Linux and MacOS).
This version supports LLVM 3.1, 3.2, 3.3 and 3.4, but now we are recommending
to avoid 3.1, as it's known to contain a number of stability problems and we are
planning to deprecate its support soon.
The released binaries are built with 3.3.

ispc 1.4.1 is Released
----------------------

A major new version of ``ispc`` has been released with stability and
performance improvements on all supported platforms (Windows, Linux and MacOS).
This version supports LLVM 3.1, 3.2, 3.3 and 3.4. The released binaries are
built with 3.2.

ispc 1.3.0 is Released
----------------------

A major new version of ``ispc`` has been released.  In addition to a number
of new language features, this release notably features initial support for
compiling to the Intel Xeon Phi (Many Integrated Core) architecture.

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
