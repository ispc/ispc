FILE=mandel

LIB=lib$(FILE)_cudart.a
all: $(LIB)


$(LIB) : $(FILE).cu
	nvcc -dc $(FILE).cu -arch=sm_35 -Xptxas=-v -dryrun 2>&1 | sed 's/\#\$$//g'|awk '{ if ($$1 == "cicc") print "cp ../__kernels.ptx " $$NF; else print $0 }' > run.sh
	sh run.sh
	nvcc -dlink -o $(FILE)_dlink.o $(FILE).o -lcudadevrt -arch=sm_35
	nvcc $(FILE).o $(FILE)_dlink.o --lib -o lib$(FILE)_cudart.a

clean: 
	/bin/rm -f *.o *.a run.sh

