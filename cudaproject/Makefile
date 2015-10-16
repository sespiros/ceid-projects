CC=nvcc
CFLAGS=-lcublas -arch=sm_13
SOURCES=main.cc cublas.cu plainKernel.cu optimizedKernel.cu 
OBJECTS=main.o cublas.o plainKernel.o optimizedKernel.o
EXECUTABLE=main

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o
