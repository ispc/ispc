This is a simple dispatch code to call ``AMD Core Math Library`` [1]_,  that has 128-bit SSE version of some trancendental that are used in ``ISPC``:

::

   sin, cos, tan, exp, log, pow

To use the library: 
 
  1. Download the library [1]_
  2. Compile ``ISPC`` program with ``--math-lib=acml`` flag
  3. Link application with ``-lamdlibm`` 


.. [1] http://developer.amd.com/tools-and-sdks/cpu-development/libm/
