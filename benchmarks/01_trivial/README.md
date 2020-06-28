# Individual language features and library functions

- ``01_aossoa`` - all variations of ``aos_to_soa``, stdlib and ISPC source implementations.
- ``02_soaaos`` - all variations of stdlib implmentation ``soa_to_aos``. TODO: ISPC source implementations.
- ``04_fastdiv`` - integer division by a constant is handled by an algorithm, which produces a code sequence without actual division operation. Current implementation relies on code generator to do the right thing on every specific platform. This benchmark tests perfomance integer division by a constant.
