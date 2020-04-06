CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops -Wno-unused-result

all: bin hin2vec

bin:
	mkdir bin

hin2vec: src/hin2vec.c
	$(CC) src/hin2vec.c -o bin/hin2vec $(CFLAGS)

clean:
	pushd bin && rm -rf hin2vec; popd
