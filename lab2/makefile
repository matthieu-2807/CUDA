CC        = nvcc
CCFLAGS   = -O3 -I/usr/local/cuda/include -Dreal=double

NVCC      = nvcc
CUFLAGS   = -arch sm_13

LINK      = nvcc
LINKFLAGS = -L/usr/local/cuda/lib64 -lcuda

all: clean myapp

myapp: main.o utils.o vecadd.o
	$(LINK) $(LINKFLAGS) utils.o main.o vecadd.o -o $@

clean:
	rm -rf myapp vecadd.linkinfo *~ *.o

main.o: main.c
	$(CC) $(CCFLAGS) -c $<

utils.o: utils.c
	$(CC) $(CCFLAGS) -c $<

vecadd.o: vecadd.cu
	$(NVCC) $(CUFLAGS) $(CCFLAGS) -c $<
