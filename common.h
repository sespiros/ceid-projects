#ifndef __COMMON__
#define __COMMON__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNIX_PATH "/tmp/ser_global.str"

/*Size of request queue*/ // standard value 50
/* ------- */#define LISTENQ  5

/* max orders issued for setting the size of shared memory shm3 and shm4*/
/* ------- */#define MAX_ORDERS LISTENQ

/* Global constants */ // standard values 3 10 10 500
/* ------- */#define NPIZZAS   3
/* ------- */#define NBAKERS   10
/* ------- */#define NDELIVERY 10
/* ------- */#define TVERYLONG 5000 //in milliseconds

/* definitions of standard times */ //standard times 100 120 150 50 100
/* ------ */int timeofPizza[]={1000,1000,1000}; //in milliseconds
/* ------ */int timeofClient[]={1000,1000};		//in milliseconds

/* Useful enums for easier access eg. timeofPizza[peperoni] */
enum pizzaTypes	{margarita, peperoni, special};
enum distanceTypes {near, far};




/* Some color codes for eye-friendly printing */
#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

/* Helper function to print fatal errors */
void fatal(char * message){
	char error_message[100];

	strcpy(error_message, "[!!] Fatal Error ");
	strncat(error_message, message, 83);
	perror(error_message);
	exit(-1);
}

/* Helper function to print debug messages */
void debug(char * message,pid_t pid){
	printf("%s[DEBUG] - %d - %s%s\n",KMAG,pid,message,KNRM);
}

#endif
