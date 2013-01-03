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
#include <time.h>
#include <signal.h>

typedef struct _order {
	int				at;
	int				prev;
	int				next;
	pthread_t		thread;
	int 			socket;
	pthread_mutex_t lock;
	pthread_cond_t 	cond;
	int 			status;
	pizzaType 		pizzas[NPIZZAS];
	distanceType	type;

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

order_info * insert (int );

void printlist(void);

struct _stack{
	int stack[MAX_ORDERS];
	int top;
	pthread_mutex_t lock;
	pthread_mutex_t stack_full;
	pthread_cond_t 	full;
};

int pull(void );
void push(int );
void printstack(void);

/* list for the pending pizzas */
list_info list;

/* stack for the free positions */
struct _stack opt;

void *order_handler(void *);

/* note to self, calculate the overhead of a single list instead of 2 in opsys12a with forks */
void *baker_thread();
void *delivery_thread();
void *delayed_thread();

int num_bakers, num_deliveries;
pthread_mutex_t baker_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t delivery_lock = PTHREAD_MUTEX_INITIALIZER;

pthread_mutex_t free_bakers_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t free_deliveries_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t free_bakers;
pthread_cond_t free_deliveries;


/* ======================================================================================
 *                                                                                 MAIN 
 * ======================================================================================
 */

static void sigInt(int sig){
	
	int next = list.head;

	while(next!=-1){
		pthread_join(list.order[next].thread, NULL);
		next = list.order[next].next;
	}


	pthread_mutex_destroy(&list.lock);
	pthread_mutex_destroy(&opt.lock);
	pthread_mutex_destroy(&opt.stack_full);

	pthread_cond_destroy(&list.cond);
	pthread_cond_destroy(&opt.full);
	pthread_cond_destroy(&free_bakers);
	pthread_cond_destroy(&free_deliveries);

	pthread_exit(NULL);
}

int main(int argc, char **argv){

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

	/*===================================================================================
	 *                                                                    INITIALIZATION
	 *===================================================================================
	 */

	list.head = -1;
	list.tail = -1;
	list.offset = 0;
	pthread_mutex_init(&list.lock, NULL);
   	pthread_cond_init(&list.cond, NULL);

	opt.top = -1;
	pthread_mutex_init(&opt.lock, NULL);
	pthread_mutex_init(&opt.stack_full, NULL);
   	pthread_cond_init(&opt.full, NULL);

   	pthread_cond_init(&free_bakers, NULL);
   	pthread_cond_init(&free_deliveries, NULL);

	num_bakers = NBAKERS;
	num_deliveries = NDELIVERY;

	/* sigint only works if the main thread is stuck in accept(no order threads) 
	 * if other threads are active sigint may be lost, pthread_sigmask is a possible workaround */
	if (signal(SIGINT, sigInt) == SIG_ERR)
		fatal("in Ctrl-C");
	
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

}

void *order_handler(void * arg){

	int i, recv_len, num_pizzas, c;
	pthread_attr_t attr;
	pthread_t bakers[NPIZZAS], delivery, delay;
	order_info *order;

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
	
	int socket = *(int *)arg;
	free(arg);

	/* ----------------------------------------------------------------------
	 *                                                       RECEIVE ORDER
	 * ----------------------------------------------------------------------
	 */

	recv_len = NPIZZAS+2;
	write(socket, "Server: Pizza Ceid, tell me your order!",40);
	if (read(socket,&buffer,recv_len)==0){	 	/* If clients disconnects before sends order */
		printf ("Rude client hanged up\n");
		exit(0);
	}

	printf("%s============== Received order %d, the order %s ================%s\n",KGRN,getpid(),buffer,KNRM);

	/* DELAYED THREAD */
	pthread_create(&delay, NULL, delayed_thread, &socket); 
	pthread_detach(delay);

	/* The order thread does all the initialization instantly and in insert waits for free space in the list (MAX_ORDERS) */
	order = insert(socket); 
	
	/* converts the buffer to int codes */
	for(i = 0;i<=strlen(buffer)-1;i++){
		order->pizzas[i] = buffer[i]-'0';
	}

	num_pizzas = strlen(buffer)-1;
	order->status = num_pizzas-1;

	order->type = buffer[strlen(buffer)-1]-'0'; 

	/* BAKER THREAD */	
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	

	for (c = num_pizzas-1; c >= 0; c--){
		
		pthread_mutex_lock(&free_bakers_lock);
		
		/* waiting for condition variables with 'while' instead of 'if' is encouraged due to spurious wakeups */
		while(num_bakers<=0){
			/*debug("Waiting for baker");*/
			pthread_cond_wait(&free_bakers,&free_bakers_lock);
			/* debug("Woke up");*/
		}

		pthread_mutex_unlock(&free_bakers_lock);

		pthread_mutex_lock(&baker_lock);
		num_bakers--;
		pthread_mutex_unlock(&baker_lock);
	
		pthread_create(&bakers[c], &attr, baker_thread, order);

	}

	for(i=num_pizzas-1;i>=0;i--){
		pthread_join(bakers[i], NULL);
	}

	debug("Finished baking");

	/* DELIVERY THREAD */	
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	pthread_mutex_lock(&free_deliveries_lock);
		
	/* waiting for condition variables with 'while' instead of 'if' is encouraged due to spurious wakeups */
	while(num_deliveries<=0){
		pthread_cond_wait(&free_deliveries,&free_deliveries_lock);
	}

	pthread_mutex_unlock(&free_deliveries_lock);

	pthread_mutex_lock(&delivery_lock);
	num_deliveries--;
	pthread_mutex_unlock(&delivery_lock);
	
	pthread_create(&delivery, &attr, delivery_thread, order);

	pthread_attr_destroy(&attr);

	pthread_join(delivery, NULL);

	debug("Finished delivering"); 
	
	/*  HERE I MUST KILL delay */
	pthread_cancel(delay);

	delete(order);
	
	write(order->socket,"DONE!\0",6);
	close(order->socket);

	pthread_exit(NULL); 
}

void *delayed_thread(void *arg){
	static struct timespec time_to_wait = {0,0};
	/*order_info *order = (order_info* ) arg;*/
	int socket = *(int*)arg;

	int j;

	time_to_wait.tv_sec = time(NULL) + TVERYLONG;

	for(j=0;;j++){
		/*  debug("Setting delay");
		pthread_mutex_lock(&order->lock);
		pthread_cond_timedwait(&order->cond, &order->lock, &time_to_wait);	
		pthread_mutex_unlock(&order->lock);*/
		sleep(TVERYLONG);

		write(socket,"Sorry for the delay you will receive a free cocacola\0",53);	
	}
}

void *baker_thread(void *arg){
	static struct timespec time_to_wait = {0,0};
	order_info *order = (order_info* ) arg;
	int type = order->pizzas[order->status];

	time_to_wait.tv_sec = time(NULL) + getPizzaTime[type]; 
	pthread_mutex_lock(&order->lock);
	pthread_cond_timedwait(&order->cond, &order->lock, &time_to_wait);	
	pthread_mutex_unlock(&order->lock);

	pthread_mutex_lock(&baker_lock);
	
	num_bakers++;

	pthread_mutex_unlock(&baker_lock);
	
	pthread_cond_signal(&free_bakers);
	
	/* pthread_exit(NULL); */
}

void *delivery_thread(void *arg){
	static struct timespec time_to_wait = {0,0};
	order_info *order = (order_info* ) arg;

	int type = order->type; 
	
	time_to_wait.tv_sec = time(NULL) + getDistanceTime[type];

	pthread_mutex_lock(&order->lock);
	pthread_cond_timedwait(&order->cond, &order->lock, &time_to_wait);	
	pthread_mutex_unlock(&order->lock);

	pthread_mutex_lock(&delivery_lock);
	
	num_deliveries++;

	pthread_mutex_unlock(&delivery_lock);
	
	pthread_cond_signal(&free_deliveries);
	
	/*  pthread_exit(NULL); */

}

void printstack(){
	int next = opt.top;
	printf ("stack :");
	while (next!=-1){
		printf("%d, ",opt.stack[next]);
		next--;
	}
	printf(" end stack \n");

}

void push(int a){

	pthread_mutex_lock(&opt.lock);	

	opt.top++;
	opt.stack[opt.top] = a;

	pthread_mutex_unlock(&opt.lock);
	
	pthread_cond_signal(&opt.full);
}

int pull(void){
	int ret;
	
	pthread_mutex_lock(&opt.lock);

	/* waiting for condition variables with 'while' instead of 'if' is encouraged due to spurious wakeups */
	while(opt.top  <= -1){
		debug("waiting");
		pthread_cond_wait(&opt.full,&opt.lock);
	}
	
	ret = opt.stack[opt.top];
	opt.top--;

		
	pthread_mutex_unlock(&opt.lock);
	
	return ret;
}

void printlist(){
	int next = list.head;
	printf ("list : \n");
	while(next!=-1){
		printf("%d <- %d -> %d\n",list.order[next].prev, list.order[next].at, list.order[next].next);
		next = list.order[next].next;
	}
	printf("end list\n");
}

int _init_proc(int sockfd) {

	pthread_t thr;
    pthread_attr_t attr;
	int *socket = malloc(sizeof(int));

	*socket = sockfd;

	/* explicitly make the thread joinable */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_create(&thr, &attr, order_handler, socket);
    pthread_attr_destroy(&attr);
	
	pthread_detach(thr);

	return 1;
}

order_info *insert(int socket){

	int offset;
	order_info *ret;

	/* insert order thread in the list */ 
	offset = list.offset;

	printf("list.offset = %d\n",offset);
	if (offset >= MAX_ORDERS){
		debug("pulling");
		offset = pull(); 
		printstack();
	}else
		list.offset++;

	pthread_mutex_lock(&list.lock);

	if (list.head == -1){		 /*  if list is empty */
		list.head = offset;
		list.order[offset].prev = -1;
	}else{
		list.order[list.tail].next = offset;
		list.order[offset].prev = list.tail;
	}
	list.tail = offset;
	list.order[offset].next = -1; 

	
	list.order[offset].at = offset;
	list.order[offset].socket = socket;
	
	/* initialize order locking mechanisms */
	pthread_mutex_init(&list.order[offset].lock, NULL);
   	pthread_cond_init(&list.order[offset].cond, NULL);


	ret = &list.order[offset];

	pthread_mutex_unlock(&list.lock);

	return ret;
}

void delete(order_info *addr){
	
	int offset = addr->at;

	pthread_mutex_lock(&list.lock);

	if (addr->prev != -1){	/* it is not the head */
		list.order[addr->prev].next = addr->next;
	}else{					/* it is the head */
		list.head=addr->next;
		if (list.head == -1) 	/* it is also the end */
			list.tail = -1;
		else
			list.order[list.head].prev=-1;
	}

	if (addr->next != -1) /* it is not the tail */
		list.order[addr->next].prev = list.order[addr->at].prev;
	
	push(offset);
	printstack();
	pthread_mutex_unlock(&list.lock);

	pthread_mutex_destroy(&addr->lock);
	pthread_cond_destroy(&addr->cond);

}
