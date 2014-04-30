
all: clean cpu gpu

cpu:
	gcc -03 sum.c -o sum_cpu

gpu:
	nvcc --ptxas-options=-v -arch sm_13 sum.cu -o sum_gpu	
	
clean:
	rm -rf sum_cpu sum_gpu *~


