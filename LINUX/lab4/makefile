
all: clean cpu gpu

cpu:
	gcc -O3 mmult.c -o mmult_cpu

gpu:
	nvcc -arch sm_20 mmult.cu -o mmult_gpu

clean:
	rm -rf mmult_cpu mmult_gpu *~

