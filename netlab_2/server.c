/*
 * =====================================================================================
 *
 *       Filename:  server.c
 *
 *    Description:  Simple TCP server for the second project networks lab
 *
 *        Version:  1.0
 *        Created:  11/25/2013 10:35:29 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Seimenis Spyros, 5070, netlab209
 *   Organization:  Ceid 2013, netlab
 *
 * =====================================================================================
 */

#include "common.h"
#include <time.h>

int sockfd, total;

int main(int argc, char **argv) {
    int i, server_port, listenfd, choice, team_offset, udp_port, sockets[LISTENQ], offset;
	socklen_t client_size;
	struct sockaddr_in server_addr, client_addr;
    struct settings udp[NUMCLIENTS];

    srand(time(NULL));
    team_offset = 9000+(209-1)*NUMCLIENTS; /* NUMCLIENTS = 10 */
    
    /* ARGUMENT SETUP */
    if(argc<3 || strcmp(argv[1],"-p")){
        printf("Usage: ./server -p <server_port>\n");
        return -1;
    } else {
        server_port = atoi(argv[2]);
        if (server_port<10080 || server_port>10089){
            printf("Please be nice and give proper port numbers between 10080 and 10089\n");
            return -1;
        }
    }

	/* ------------------------------------- SOCKET SETUP ------------------------ */
    if ((listenfd = socket(AF_INET, SOCK_STREAM, 0)) == -1)
        fatal("in socket");

    bzero(&server_addr, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(server_port);
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(listenfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1)
        fatal("binding to socket");

    if (listen(listenfd, LISTENQ) == -1)
        fatal("listening to socket");

	total = 0;
    while(1){
		/* The incoming connections are processed one by one
           No need for fork because of the simplicity of the process
		*/
		
	    /* ----- ESTABLISH A CONNECTION BETWEEN 2 CLIENTS ----------------- */		
		client_size = sizeof(client_addr);
		sockfd = accept(listenfd, (struct sockaddr*)&client_addr, &client_size);

		if(sockfd < 0){
			if (errno == EINTR )continue;
			else fatal("in accepting connection");
		}

		printf("Client %d connected\n",total);
       
        /* Read the udp port that the clients will use for their connection */
        read(sockfd, &udp_port, sizeof(udp_port));

        offset = udp_port-team_offset;

        /* If there exists a client using the same udp port, connect them */
        if (udp_port == udp[offset].udp_port){
		    i=1;
            /* inform clients about the state of the connection */
            write(sockfd,&i,sizeof(i)); 

            if(udp[offset].type==1){
                udp[offset].client_addr = client_addr;
		        udp[offset].client_size = sizeof(client_addr);
            }else{
		        udp[offset].server_addr = client_addr;
		        udp[offset].server_size = sizeof(client_addr);
            }

            if(udp[offset].type==1)udp[offset].type ^=1; 
            write(sockets[offset],&udp[offset],sizeof(udp[offset]));
            udp[offset].type ^= 1;
            write(sockfd,&udp[offset],sizeof(udp[offset]));

            /* clean the static table so the next client using the same port can used */
            udp[offset].udp_port = 0;

            close(sockets[offset]);
            close(sockfd);

        }else{
            udp[offset].udp_port = udp_port;
		    /* randomly select the types, 0 means client serves as udp server */
		    choice = rand()%2;

            i=0;
            /* inform clients about the state of the connection */
            write(sockfd,&i,sizeof(i)); 

		    /* 
		    The same struct is sent to both clients 
		    with only difference the udp.type option 
		    for server/client
		    */
            if (choice==0) {
		        udp[offset].server_addr = client_addr;
		        udp[offset].server_size = sizeof(client_addr);
            }else{
                udp[offset].client_addr = client_addr;
		        udp[offset].client_size = sizeof(client_addr);
            }
		    udp[offset].type = choice;
            sockets[offset] = sockfd;
            
        }

		

		total+=1;
    }
}
