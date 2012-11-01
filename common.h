#ifndef __COMMON__
#define __COMMON__
#include <stdlib.h>
#include <string.h>


#define UNIX_PATH "/tmp/ser_global.str"

/* Useful enums for easier access eg. timeofPizza[peperoni] */
enum pizzaTypes	{margarita, peperoni, special};
enum distanceTypes {near, far};

/* Global constants */
#define NPIZZAS   3
#define NBAKERS   10
#define NDELIVERY 10
#define TVERYLONG 50

#define SHMSZ 16
#define SHMGLOBAL 5070

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

#endif
