all: clean build

build:
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -rdc=true -o libKernel.so --shared project.cu

run:
	./project

clean:
	rm ./libKernel.so

lib:
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -o libKernel.so --shared project.cu
