/*
 * =====================================================================================
 *
 *       Filename:  server.c
 *		  Project:  Operating Systems I - Project 1 - 5th Semester 2012
 *    Description:  Accepts orders
 *
 *        Version:  1.0
 *        Created:  10/30/2012 02:14:20 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Seimenis Spiros 5070 
 *   Organization:  Computer Engineering and Informatics Department
 *                  University of Patras
 *
 * =====================================================================================
  */
#include "common.h"
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>

#include <errno.h>
#include <sys/wait.h>
#include <sys/un.h>

/*Size of request queue*/
#define LISTENQ  20


/*  Helper function to avoid zombie processes */
void sig_chld(int signo);

int main(int argc, char **argv){
	
	/* definitions of standard times (used better with the enums defined in common.h)*/	
	int timeofPizza[]={10,12,15};
	int timeofClient[]={5,10};

	/* holds the client order information in the following way
	 *      |-----------------------------------------------|
	 * 		|pizza #1 | pizza #2 | pizza #NPIZZA | near/far |
	 *      |-----------------------------------------------|
	 *		to order 2 peperonis, 1 special for far is 1|1|2|1
	 *		according to the enum defined in common.h
	 */
	char buffer[NPIZZAS+1];

	int recv_length=1;

	/* Standard socket creation */
	int listenfd, sockfd;
	pid_t childpid;
	socklen_t client_size;
	struct sockaddr_un server_addr, client_addr;

	signal(SIGCHLD, sig_chld); 	/* sigchld handler to avoid zombie process generation*/
	unlink(UNIX_PATH);			/* deletes the specified file from disk to use as socket */

	if ((listenfd = socket(AF_LOCAL, SOCK_STREAM, 0))==-1)
		fatal("in socket");

	bzero(&server_addr,sizeof(server_addr));
	server_addr.sun_family = AF_LOCAL;
	strcpy(server_addr.sun_path, UNIX_PATH);
	
	if (bind(listenfd,(struct sockaddr*)&server_addr,sizeof(server_addr)) == -1 )
		fatal("binding to socket");

	/* begin listening to socket a.k.a begin taking orders from clients */
	if (listen(listenfd, LISTENQ)==-1)
		fatal("listening to socket");

	while(1){
		client_size = sizeof(client_addr);

		sockfd = accept(listenfd, (struct sockaddr*)&client_addr,&client_size);
		if(sockfd < 0 ){
			if (errno == EINTR ) /* Interrupt received */
				continue;
			else
				fatal("in accepting connection");
		}

		/* fork to handle an order exclusively */
		childpid=fork();

		if(childpid==0){ /* child process */
			close(listenfd);/* no reason to continue listening for orders */
			send(sockfd, "Server: Pizza Ceid, tell me your order!\n",39,0);
			recv(sockfd,&buffer,(NPIZZAS+1)*sizeof(char),0);
			printf("%s\n",&buffer);
			

			exit(0);
		}
		close(sockfd);


	}
}

void sig_chld( int signo) {
       pid_t pid;
       int stat;

       while ( ( pid = waitpid( -1, &stat, WNOHANG ) ) > 0 ) {
              printf( "Child %d terminated.\n", pid );
        }
 }


