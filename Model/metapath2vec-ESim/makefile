CC = g++
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lm -lgslcblas
LFLAGS = -lgsl -lm -lgslcblas
INCLUDES = -I/usr/local/include -Ieigen-3.3.3
LIBS = -L/usr/local/lib

BIN = bin/esim

all: bin $(BIN)

bin:
	mkdir bin

bin/esim : bin/ransampl.o bin/linelib.o bin/main.o bin
	$(CC) $(CFLAGS) -o bin/esim bin/ransampl.o bin/linelib.o bin/main.o $(INCLUDES) $(LIBS) $(LFLAGS)

bin/ransampl.o : src/ransampl.c bin
	$(CC) $(CFLAGS) -c src/ransampl.c $(INCLUDES) $(LIBS) $(LFLAGS) -o bin/ransampl.o

bin/linelib.o : src/linelib.cpp src/ransampl.h bin
	$(CC) $(CFLAGS) -c src/linelib.cpp $(INCLUDES) $(LIBS) $(LFLAGS) -o bin/linelib.o

bin/main.o : src/main.cpp bin/linelib.o bin
	$(CC) $(CFLAGS) -c src/main.cpp $(INCLUDES) $(LIBS) $(LFLAGS) -o bin/main.o

clean :
	rm -rf bin
