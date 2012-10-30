CC = gcc
CFLAGS = 

server: server.c
	$(CC) -o server server.c 

client: client.c
	$(CC) -o client client.c 
