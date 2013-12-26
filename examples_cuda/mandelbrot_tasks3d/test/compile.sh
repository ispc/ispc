#!/bin/sh
nvcc -arch=sm_35 -dc kernel_ptx.cu -dryrun -Xptxas=-v 2>&1  | \
 sed 's/\#\$//g'| \
 awk '{if ($1=="cicc") {print $0;  print "grep -ve \"\\.version\" -e \"\\.target\" -e \"\\.address_size\" ", $NF, " > __body.ptx"; print "cat __header.ptx __body.ptx >", $NF} else print $0}' > run1.sh
