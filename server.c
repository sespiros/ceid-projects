/*
 * =====================================================================================
 *
 *       Filename:  server.c
 *		  Project:  Operating Systems I - Project 2 - 5th Semester 2012
 *    Description:  Accepts orders.
 *
 *        Version:  1.0
 *        Created:  29/12/2012 21:41:33 AM
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
#include <pthread.h>

typedef struct _order {
	int				at;
	int				prev;
	int				next;
	pthread_t		thread;
	int 			socket;
	pthread_mutex_t lock;
	pthread_cond_t 	cond;

}	order_info;

typedef struct _list{
	order_info 	order[MAX_ORDERS];
	int			head;
	int 		tail;
	int 		offset;
	pthread_mutex_t lock;
	pthread_cond_t 	cond;
	
} list_info;

/* Creates an order_info, inserts it in the given list */
int _init_proc_(int);

/* Deletes a order_info node from the given list */
void delete(order_info*);

void printlist(void);

struct _stack{
	int stack[MAX_ORDERS];
	int top;
	pthread_mutex_t lock;
	pthread_cond_t 	full;
	pthread_cond_t 	empty;
};

int pull(void );
void push(int );
void printstack(void);

/* list for the pending pizzas */
list_info list;

/* stack for the free positions and mutex for controlling it*/
struct _stack opt;

void *order_handler(void *);

/* ======================================================================================
 *                                                                                 MAIN 
 * ======================================================================================
 */

int main(int argc, char **argv){

	/*===================================================================================
	 *                                                                    INITIALIZATION
	 *===================================================================================
	 */

	list.head = -1;
	list.tail = -1;
	list.offset = -1;
	pthread_mutex_init(&list.lock, NULL);
   	pthread_cond_init(&list.cond, NULL);

	opt.top = -1;
	pthread_mutex_init(&opt.lock, NULL);
   	pthread_cond_init(&opt.full, NULL);
   	pthread_cond_init(&opt.empty, NULL);
	
	/*===================================================================================
	 *                                                                    SOCKETS SETUP
	 *===================================================================================
	 */
	int sockfd, listenfd;
	pid_t childpid;
	socklen_t client_size;
	struct sockaddr_un server_addr, client_addr;

	unlink(UNIX_PATH);			/* deletes the specified file from disk to use as socket */

	if ((listenfd = socket(AF_LOCAL, SOCK_STREAM, 0))==-1)
		fatal("in socket");

	bzero(&server_addr,sizeof(server_addr));
	server_addr.sun_family = AF_LOCAL;
	strcpy(server_addr.sun_path, UNIX_PATH);
	
	if (bind(listenfd,(struct sockaddr*)&server_addr,sizeof(server_addr)) == -1 )
		fatal("binding to socket");


	/*-----------------------------------------------------------------------------------
	 * begin listening to socket a.k.a begin taking orders from clients 
	 *-----------------------------------------------------------------------------------
	 */
	if (listen(listenfd, LISTENQ)==-1)
		fatal("listening to socket");

	/* ----------------------------------------------------------------------------------
	 * Server starts and accepts connections
	 * ----------------------------------------------------------------------------------
	 */
	while(1){
		client_size = sizeof(client_addr);

		sockfd = accept(listenfd, (struct sockaddr*)&client_addr,&client_size);

		if(sockfd < 0 ){
			if (errno == EINTR ) 		/* If interrupt received the system call continues normally*/
				continue;
			else
				fatal("in accepting connection");
		}

		/* new thread to handle an order exclusively */
		if (!_init_proc(sockfd))
			fatal("in initializing client thread");

		
	}

	pthread_mutex_destroy(&list.lock);
	pthread_cond_destroy(&list.cond);
	pthread_mutex_destroy(&opt.lock);
	pthread_cond_destroy(&opt.full);
	pthread_cond_destroy(&opt.empty);
}


void *order_handler(void * arg){

	int i, recv_len;
	pizzaType pizzas[NPIZZAS];
	distanceType type;

	// addr is the address of the struct for the specific order
	order_info *order = (order_info *)arg;

	/* ------------------------------------------------------------
	 * holds the client order information in the following way
	 *      |=====================...==========================|
	 * 		|pizza #1 | pizza #2 |...pizza #NPIZZA |  near/far |
	 *      |=====================...==========================|
	 *		to order 2 peperonis, 1 special for far is 1|1|2|1
	 *		according to the enum defined in common.h
	 * ------------------------------------------------------------
	 */
	char buffer[NPIZZAS+1];

	/* ----------------------------------------------------------------------
	 *                                                       RECEIVE ORDER
	 * ----------------------------------------------------------------------
	 */

	recv_len = NPIZZAS+2;
	write(order->socket, "Server: Pizza Ceid, tell me your order!",40);
	if (read(order->socket,&buffer,recv_len)==0){	 	/* If clients disconnects before sends order */
		printf ("Rude client hanged up\n");
		exit(0);
	}

	printf("%s============== Received order %d, the order %s ================%s\n",KGRN,getpid(),buffer,KNRM);

	/* converts the buffer to int codes */
	for(i = 0;i<strlen(buffer)-1;i++){
		pizzas[i] = buffer[i]-'0';
	}

	type = buffer[strlen(buffer)-1]-'0';

	printlist();
	printstack();

	delete(order);

	printlist();
	printstack();

	pthread_exit(NULL);
}
void printstack(){
	printf ("stack :\n");
	int next = opt.top;
	while (next!=-1){
		printf("%d, ",opt.stack[next]);
		next--;
	}
	printf("\n");

}

void push(int a){
	pthread_mutex_lock(&opt.lock);	
	
	while(opt.top + 1 >= MAX_ORDERS)
		pthread_cond_wait(&opt.empty,&opt.lock);
	
	opt.top++;
	opt.stack[opt.top] = a;

	pthread_cond_signal(&opt.full);
	pthread_mutex_unlock(&opt.lock);
}

int pull(void){
	int tmp;
	
	pthread_mutex_lock(&opt.lock);

	while(opt.top == -1)
		pthread_cond_wait(&opt.full,&opt.lock);
	
	tmp = opt.top;
	opt.top--;
		
	pthread_cond_signal(&opt.empty);
	pthread_mutex_unlock(&opt.lock);

	return opt.stack[tmp];
}

void printlist(){
	printf ("list :\n");
	int next = list.head;
	while(next!=-1){
		printf("%d ",list.order[next].at);
		next = list.order[next].next;
	}
	printf("\n");
}


int _init_proc(int sockfd) {

	int offset;

	/* explicitly make the thread joinable */
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	/* insert order thread in the list */ 
	if (list.offset>=MAX_ORDERS){
		offset = pull(); 
	}else
		offset = ++list.offset;
  
	pthread_mutex_lock(&list.lock);

	if (list.head == -1){		/* if list is empty */
		list.head = offset;
		list.order[offset].prev = -1;
	}else{
		list.order[list.tail].next = offset;
		list.order[offset].prev = list.tail;
	}
	list.tail = offset;
	list.order[list.tail].next = -1;

	pthread_mutex_unlock(&list.lock);
	
	list.order[offset].at = offset;
	list.order[offset].socket = sockfd;

	// initialize order locking mechanisms
	pthread_mutex_init(&list.order[offset].lock, NULL);
   	pthread_cond_init(&list.order[offset].cond, NULL);
   
    pthread_create(&list.order[offset].thread, &attr, order_handler, &list.order[offset]);
    pthread_attr_destroy(&attr);

	return 1;
}

void delete(order_info *addr){

	pthread_mutex_lock(&list.lock);

	if (addr->prev != -1)	/* it is not the head */
		list.order[addr->prev].next = addr->next;
	else{					/* it is the head */
		list.head=addr->next;
		if (list.head == -1) 	/* it is also the end */
			list.tail = -1;
		else
			list.order[list.head].prev=-1;
	}

	int offset = addr->at;

	pthread_mutex_unlock(&list.lock);
	
	push(offset);

	pthread_mutex_destroy(&addr->lock);
	pthread_cond_destroy(&addr->cond);

}
