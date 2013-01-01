#ifndef __COMMON__
#define __COMMON__

/* if _DEBUG_ is defined the output of the server will be veeeeery verbose */
/* #define _DEBUG_ */

#include <sys/types.h>		/* Type definitions */
#include <stdio.h> 			/* standard I/O */
#include <stdlib.h>			/* Prototypes of commonly used library functions */

#include <unistd.h>			/* Prototypes of many system calls */
#include <errno.h>			/* Declares errno and defines error constants */
#include <string.h>			/* Commonly used string-handling functions */

#include <sys/socket.h> 	/* socket library */
#include <sys/un.h>			/* Definition for UNIX domain sockets*/

#define UNIX_PATH "/tmp/ser_global.str" /* for UNIX domain socket */

/*Size of request queue*/ 
#define LISTENQ  10
/* didnt work with small listenq and 4 and 10 orders ERROR CHECK IT */

/* max orders issued for setting the size of shared memory lists*/
#define MAX_ORDERS LISTENQ

/* Global constants */ 
#define NPIZZAS   3
#define NBAKERS   10
#define NDELIVERY 10
#define TVERYLONG 20L					/* in seconds */
	
/* definitions of standard times */ 
long int getPizzaTime[]={1L,5L,3L}; 		/* in seconds */
long int getDistanceTime[]={3L,5L};			/* in seconds */

/* typedef pizzaType and distanceType for reference */
typedef enum { margarita, peperoni, special } pizzaType;
typedef enum { near, far } distanceType;

/* Some color codes for eye-friendly printing */
#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

/* Simple function to print fatal errors */
void fatal(char * message){
	char error_message[100];

	strcpy(error_message, "[!!] Fatal Error ");
	strncat(error_message, message, 83);
	perror(error_message);
	exit(-1);
}

/* Simple function to print debug messages */
void debug(char * message){
	printf("%s[DEBUG] - %s%s\n",KMAG,message,KNRM);
}

#endif
