# Règle principale
all: opencv_test.pgr

# Variables de compilation
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -O3
LDFLAGS =

# Flags spécifiques à OpenCV
CFLAGS += $(shell pkg-config --cflags opencv4)
LDFLAGS += $(shell pkg-config --libs opencv4)

# Flags pour OpenMP et CUDA
CFLAGS += -fopenmp
LDFLAGS += -fopenmp -L/usr/local/cuda/lib64 -lcuda
# Flag for a GTX 1070
# For ROMEO, use -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_60
NVCCFLAGS = -Xcompiler -fopenmp -gencode arch=compute_61,code=sm_61

# Règles de compilation
opencv_test.pgr: obj/opencv_simple.o obj/gaussianNoise.o obj/jacobi.o obj/gauss_seidel.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

obj/%.o: %.cpp
	mkdir -p obj
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

obj/%.o: %.cu
	mkdir -p obj
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(LDFLAGS)

# Règle pour nettoyer les fichiers objets
proper:
	rm -f obj/*.o

# Règle pour nettoyer tous les fichiers générés
clean:
	rm -rf obj/*.o *.pgr
