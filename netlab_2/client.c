/*
 * =====================================================================================
 *
 *       Filename:  client.c
 *
 *    Description:  Simple TCP/UDP client/server for the second project networks lab
 *
 *        Version:  1.0
 *        Created:  11/26/2013 12:18:45 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Seimenis Spyros, 5070, netlab209
 *   Organization:  Ceid 2013, netlab
 *
 * =====================================================================================
 */

#include "common.h"

int main(int argc, char **argv){
    int listenfd, sockfd, tcp_port, udp_port, pid, order, n;
    in_addr_t server_ip;
    struct sockaddr_in server_addr;
    char buff[1000];
    struct settings udp;

	/* ARGUMENT SETUP */
    if (argc<5 || strcmp(argv[1],"-s")){
        printf("Usage: ./client -s <server_ip> <tcp_port> <udp_port>\n");
        return -1;
    }else{
        server_ip = inet_addr(argv[2]);
        tcp_port = atoi(argv[3]);
        udp_port = atoi(argv[4]);

        if (tcp_port<10080||udp_port<10080||tcp_port>10089||udp_port>10089){/* ports between 10080 and 10089 based on team number */
            printf("Please be nice and give proper port numbers between 0 and 65535 for tcp server port and 10080-10089 for udp \n");
            return -1;
        }
    }

	/* -------- SOCKET SETUP ------------------------------------------- */
    sockfd = socket(AF_INET, SOCK_STREAM, 0);

    bzero(&server_addr, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(tcp_port);
    server_addr.sin_addr.s_addr = server_ip;

    if(connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1)
        fatal("in connection to server, maybe server is not up");

	/* -------- ESTABLISH THE CONNECTION ------------------------------- */	
    printf("%s=================== TCP/UDP CLIENT ===================\n", KBLU);
    printf("|%s[+]Connection to server established%s                 |\n", KYEL, KBLU);
    /* Send the udp port information to server for sharing with the udp clients */
    write(sockfd, &udp_port, sizeof(udp_port));
	/* gets information to find out which client arrived first,
    useful only for the first client arrived */
    read(sockfd, &order, sizeof(order));
    if(order==0)
        printf("%s|%s[+]Waiting for second client...%s                     |\n", KBLU, KYEL, KBLU);

	/* gets the settings struct used for the udp connection setup */
    read(sockfd, &udp, sizeof(udp));

    if(order==0)
        printf("%s|%s[+]Second client joined the room%s                    |\n", KBLU, KYEL, KBLU); 
    else
        printf("%s|%s[+]Route to client 1 established%s                    | \n", KBLU, KYEL, KBLU); 

	close(sockfd);
	
    /* -------- SETUP THE UDP CONNECTION -------------------------------- */
    if ((listenfd = socket(AF_INET, SOCK_DGRAM, 0)) == -1)
        fatal("in udp socket");

    udp.server_addr.sin_port = htons(udp_port);

    udp.client_addr.sin_port = htons(udp_port+1);

    printf("%s======================================================\n",KBLU);
    if(udp.type==0)
    {
        printf("%s[+]Client serving as server waiting message....\n", KMAG);
        
        if (bind(listenfd, (struct sockaddr*)&udp.server_addr, sizeof(udp.server_addr)) == -1)
            fatal("binding to udp socket");

        while(1){
            n = recvfrom(listenfd, buff, 1000, 0, (struct sockaddr*)&udp.client_addr, &udp.client_size);
            buff[n] = 0;
            printf("%sCLIENT:%s", KGRN, buff);
            
            printf("%sSERVER:", KRED);
            fgets(buff, 1000, stdin);

            sendto(listenfd, buff, 1000, 0, (struct sockaddr*)&udp.client_addr, sizeof(udp.client_addr));
        }
    }else{
        /* binding on the udp "client" makes him actually a udp "server" and allows
           for duplex communication, comment if only one side communication is required */
        if (bind(listenfd, (struct sockaddr*)&udp.client_addr, sizeof(udp.client_addr)) == -1)
            fatal("binding to udp socket");

        printf("%sClient serving as client send message:\n", KMAG);
        while(1){
            printf("%sCLIENT:", KGRN);
            fgets(buff, 1000, stdin);
            sendto(listenfd, buff, strlen(buff), 0, (struct sockaddr*)&udp.server_addr, sizeof(udp.server_addr));
        
            n = recvfrom(listenfd, buff, 1000, 0, (struct sockaddr*)&udp.server_addr, &udp.server_size);
            buff[n] = 0;
            printf("%sSERVER:%s%s", KRED, buff, KNRM);
        }
    }

    return 0;
}
