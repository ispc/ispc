# Individual language features and library functions

- ``01_aossoa`` - all variations of ``aos_to_soa``, stdlib and ISPC source implementations.
- ``02_soaaos`` - all variations of stdlib implmentation ``soa_to_aos``. TODO: ISPC source implementations.
- ``03_popcnt`` - test ``popcnt()`` stdlib function perfomance.
- ``04_fastdiv`` - integer division by a constant is handled by an algorithm, which produces a code sequence without actual division operation. Current implementation relies on code generator to do the right thing on every specific platform. This benchmark tests perfomance integer division by a constant.
- ``05_packed_load_store`` - test ``packed_[load|store]_active()`` stdlib functions perfomance.
