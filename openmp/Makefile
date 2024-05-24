all: opencv_test.pgr

CC=g++
CFLAGS=-Wall -O3
LDFLAGS=

CFLAGS += $(shell pkg-config --cflags opencv4) -fopenmp
LDFLAGS += $(shell pkg-config --libs opencv4) -fopenmp

opencv_test.pgr: obj/opencv_simple.o obj/gaussianNoise.o
	$(CC) $(CFLAGS) $^ -o $@  $(LDFLAGS)

obj/%.o: %.cpp
	mkdir -p obj
	$(CC) $(CFLAGS) $< -c -o $@

proper :
	rm -f obj/*.o

clean:
	rm -rf obj/*.o *.pgr
