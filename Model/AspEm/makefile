CC = g++
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lm -lgslcblas
LFLAGS = -lgsl -lm -lgslcblas
INCLUDES = -I/usr/local/include -Ieigen-3.3.3
LIBS = -L/usr/local/lib
BIN = bin/aspem

all: bin ${BIN}

bin:
	mkdir bin

bin/aspem : bin/linelib.o bin/main.o bin
	$(CC) $(CFLAGS) -o bin/aspem bin/linelib.o bin/main.o $(INCLUDES) $(LIBS) $(LFLAGS)

bin/linelib.o : src/linelib.cpp bin
	$(CC) $(CFLAGS) -c src/linelib.cpp $(INCLUDES) $(LIBS) $(LFLAGS) -o bin/linelib.o

bin/main.o : src/main.cpp bin/linelib.o bin
	$(CC) $(CFLAGS) -c src/main.cpp $(INCLUDES) $(LIBS) $(LFLAGS) -o bin/main.o

clean :
	rm -rf bin

