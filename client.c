/*
 * =====================================================================================
 *
 *       Filename:  client.c
 *  	  Project:  Operating Systems I - Project 1 - 5th Semester 2012
 *    Description:  The client is responsible for delivering an order to the server
 *          Usage:  Calling ./client prompts user for input 
 *          		Calling ./client with an argument creates random order
 *
 *        Version:  1.1
 *        Created:  10/30/2012 04:36:15 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Spyros Seimenis 5070 
 *   Organization:  Computer Engineering and Informatics Department
 *                  University of Patras 
 *
 * =====================================================================================
*/
#include "common.h"

int read_order(char *, int );

int main(int argc, char **argv){
	int sockfd;
	struct sockaddr_un server_addr;
	int pid;
	char buffer[NPIZZAS+1],greeting[40],response[53];
	if (argc<=1)printf("Calling Ceid Pizzeria. Terminate call with Ctrl-C.\n");

	sockfd = socket (AF_LOCAL,SOCK_STREAM,0);

	bzero(&server_addr, sizeof(server_addr));
	server_addr.sun_family = AF_LOCAL;
	strcpy(server_addr.sun_path, UNIX_PATH);

	if(connect(sockfd, (struct sockaddr*)&server_addr,sizeof(server_addr))==-1)
		fatal("in connection to server, maybe server is not up");
	
	read(sockfd,&greeting,40);
	if(argc<=1) printf("%s\n",greeting);

	/* The read_order function will prompt user for input if argc == 1 else will create random order */
	read_order(buffer,argc);

	/* Send order to server */
	write(sockfd,buffer,(NPIZZAS+2));

	/* Wait for server to send done or coca collas messages */
	while(1){
		if (read(sockfd,&response,53)==0)exit(0);
		printf("Server to Client %d: %s \n",getpid(),response);
		if (strcmp(response,"DONE!")==0){
			printf("Client %d closes\n",getpid());
			exit(0);
		}
		bzero(&response,53);
	}
}

int read_order(char *buffer, int flags){
	if (flags>1){	/* if ./client has arguments creates random order */
		/* initialize random number generation with a finer time seed */
		struct timeval time;
		gettimeofday(&time,NULL);
		srand((time.tv_sec*1000)+(time.tv_usec/1000));
		int i;
		for(i=0;i<(rand()%3)+1;i++)
		//for(i=0;i<2;i++)
			buffer[i]='0'+rand()%3;
		buffer[i++]='0'+rand()%2;
		buffer[i]='\0';
	}else{			/*  if ./client has no arguments prompts the user for input */
		int done=0;
		printf("[%d] for margarita\t[%d]near\n[%d] for peperoni\t[%d]far\n[%d] for special\n",margarita,near,peperoni,far,special);
		printf("Place your order under %d pizzas and 1 char for distance:%s(eg.1221 1 marg, 2 peperoni, far)%s\n ",NPIZZAS,KBLU,KNRM);
		while(!done){
			scanf("%s",buffer);
			getchar();
			strncpy(buffer,buffer,NPIZZAS+1);
			buffer[NPIZZAS+1]='\0';
			if(!(buffer[strlen(buffer)-1]=='0' || buffer[strlen(buffer)-1]=='1'))
				printf("%s[!!]Wrong distance given(last byte), only 0 or 1 accepted%s\n",KRED,KNRM);
			else{
				printf("%s[!]Warning:Input will truncate to the first %d chars%s\n",KYEL,NPIZZAS+1,KNRM);		
				done=1;
			}
		}
	}	
}	

