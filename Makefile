CC = gcc
CFLAGS = -lpthread -g -pedantic
DEPS = common.h 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

all: server client

server: server.o
	$(CC) -o $@ $^ $(CFLAGS)

client: client.o
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm *.o

