#ifndef __COMMON__
#define __COMMON__

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <sys/socket.h> /* socket library */
#include <netinet/in.h> /* Definition for network domain sockets */ 
#include <arpa/inet.h>

/* Some color codes for eye-friendly printing */
#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define LISTENQ 20 
#define NUMCLIENTS 10

/* Simple function for printing fatal errors */
void fatal(char * message) {
    char error_message[100];
    
    strcpy(error_message, "[!!] Fatal Error ");
    strncat(error_message, message, 83);
    perror(error_message);
    exit(-1);
}

/* Simple struct to share udp connection data between clients */
struct settings
{
    int type;
    int udp_port;
    socklen_t client_size, server_size;
    struct sockaddr_in server_addr, client_addr;
};

#endif
