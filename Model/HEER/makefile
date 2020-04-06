CC = g++
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result 

all: bin bin/line

bin:
	mkdir bin

bin/line : bin/ransampl.o src/line.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) -I/home/yuxinx2/gsl/include -L/home/yuxinx2/gsl/lib src/line.cpp bin/ransampl.o -o bin/line -lgsl -lm -lgslcblas

bin/ransampl.o : src/ransampl.c
	$(CC) $(CFLAGS) $(OPT_DEF) -c src/ransampl.c -o bin/ransampl.o

clean:
	rm -rf bin
