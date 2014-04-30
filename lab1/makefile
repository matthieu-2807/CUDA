
all: clean cpu gpu

cpu:
	gcc -O3 vecadd.c -o vecadd_cpu

gpu:
	nvcc vecadd.cu -o vecadd_gpu

clean:
	rm -rf vecadd_cpu vecadd_gpu vecadd.linkinfo *~
