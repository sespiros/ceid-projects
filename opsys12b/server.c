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

/* -------------------------------------------------   STRUCTS  -----------
 * =======================================================================
 * ----------------------------------------------------------------------- */

/* order_info is the node used for the order list
 * */
typedef struct _order {
	/* the thread that handles this order */
	pthread_t		thread;

	/* variables for handling the double linked list of orders */
	int				at;
	int				prev;
	int				next;

	/* Optional variable that shows how many pizzas are left to be completed */
	int 			status;

	/* The types of pizzas and the distance */
	pizzaType 		pizzas[NPIZZAS];
	distanceType	type;

}order_info;

/* list_info is the type of the double linked list i use implemented on a table
 * */
typedef struct _list{
	/* a fixed size order_info table, head and tail of the list */
	order_info 	order[MAX_ORDERS];
	int			head;
	int 		tail;

	/* The offset value shows the next free position in the order table
	 * when this offset exceeds MAX_ORDERS limit is not used anymore because
	 * the nodes take their position from the stack of free positions */
	int 		offset;

	/* The condition variable and its mutex to lock the list when inserting or deleting */
	pthread_mutex_t lock;
	pthread_cond_t 	cond;

} list_info;

/* the type of the stack of free positions implemented on table
 * */
struct _stack{
	/* fixed size table of free_positions */
	int stack[MAX_ORDERS];
	int top;

	/* locks the stack when read/write */
	pthread_mutex_t lock;

	/* signals when an item exists in the stack (after pushing) */
	pthread_cond_t 	full;
};

/* -------------------------------------------------  FUNCTIONS  -----------
 * ========================================================================
 * ----------------------------------------------------------------------- */

/* Initializes the order thread when it arrives on the server */
int _init_proc_(int);

/* Deletes a order_info node from the list */
void delete(order_info*);

/* Inserts an order_info node in the list and returns a pointer to it */
order_info * insert (pthread_t );

/* Self explanatory */
void printlist(void);

/* Returns a free position in the list.order table */
int pull(void );

/* Pushes a free position in the list.order table */
void push(int );

/* no */
void printstack(void);

/* --------------------------------------------   THREAD FUNCTIONS ----------
 * *************************************************************************
 * ------------------------------------------------------------------------*/

/* The order thread */
void *order_handler(void *);

/* the baker thread, each order thread throws as much as needed controlled by num_bakers */
void *baker_thread();

/* the delivery thread, same as baker thread */
void *delivery_thread();

/* each order thread throws a delayed thread that sends a message to each client every TVERYLONG */
void *delayed_thread();

/* num_bakers holds the number of available bakers each moment
 * every order thread checks this before throwing bakers
 * controlled by baker_lock. same apply for the num_deliveries*/
int num_bakers, num_deliveries;
pthread_mutex_t baker_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t delivery_lock = PTHREAD_MUTEX_INITIALIZER;

/* When a baker is done baking and increments num_bakers signals the free_bakers to wakeup
 * sleeping order threads waiting for bakers, same for deliveries */
pthread_cond_t free_bakers;
pthread_cond_t free_deliveries;


/* list for the pending pizzas */
list_info list;

/* stack for the free positions */
struct _stack opt;

/* ======================================================================================
 *                                                                                 MAIN
 * ======================================================================================
 */

int main(int argc, char **argv){

	/*===================================================================================
	 *                                                                    SOCKETS SETUP
	 *===================================================================================
	 */
	int sockfd, listenfd, i;
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

	/* list initialization */
	list.head = -1;
	list.tail = -1;
	list.offset = 0;
	pthread_mutex_init(&list.lock, NULL);
   	pthread_cond_init(&list.cond, NULL);

	/* stack initialization */
	opt.top = -1;
	pthread_mutex_init(&opt.lock, NULL);
   	pthread_cond_init(&opt.full, NULL);

	num_bakers = NBAKERS;
	num_deliveries = NDELIVERY;

	pthread_cond_init(&free_bakers, NULL);
   	pthread_cond_init(&free_deliveries, NULL);



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

/* Initializes the thread */
int _init_proc(int sockfd) {

	pthread_t thr;
    pthread_attr_t attr;

	/* Creation of a heap variable socket that passes to the order thread and is freed there
	 * so that there is no conflict between the socket THIS order received and the socket of the next orders */
	int *socket = malloc(sizeof(int));

	*socket = sockfd;

	/* explicitly make the thread joinable */
    pthread_attr_init(&attr);
	/* DONT KNOW IF THIS IS NEEDED. CHECK IT*/
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_create(&thr, &attr, order_handler, socket);
    pthread_attr_destroy(&attr);

	/* CHECK */
	pthread_detach(thr);

	return 1;
}

/* The order thread */
void *order_handler(void * arg){

	int i, recv_len, num_pizzas, c;
	pthread_attr_t attr;

	/* Each order thread throws the bakers that are needed, the delivery and the delay thread */
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

	/* Create a new variable to hold the socket received from the _init_proc and then free it */
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

	/* After accepting the order starts the thread that count to TVERYLONG */
	pthread_create(&delay, NULL, delayed_thread, &socket);
	pthread_detach(delay);

	/* After the order thread is initialized, tries to insert itself in the list */
	order = insert(pthread_self());

	/* converts the buffer to int codes */
	for(i = 0;i<=strlen(buffer)-1;i++){
		order->pizzas[i] = buffer[i]-'0';
	}

	/* Number of pizzas to bake */
	num_pizzas = strlen(buffer)-1;

	/* The status of the order (MAYBE OPTIONAL)*/
	order->status = num_pizzas;

	/* The last char of the buffer shows the type of distance near/far */
	order->type = buffer[strlen(buffer)-1]-'0';

	/* ------------------------------------------------------- BAKER THREAD ------------------
	 * --------------------------------------------------------------------------------------*/
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	/* With this loop the order thread tries to apply its pizzas to available bakers */
	for (c = num_pizzas-1; c >= 0; c--){

		/* Lock the baker_lock in order to check for free bakers and decrement its value */
		pthread_mutex_lock(&baker_lock);

		/* waiting for condition variables with 'while' instead of 'if' is encouraged due to spurious wakeups */
		while(num_bakers<=0){
			/*debug("Waiting for baker");*/

			/* if no free bakers are available this thread waits HERE
			 * when in cond_wait the baker_lock is unlocked and other orders
			 * can proceed with checking for available bakers*/
			pthread_cond_wait(&free_bakers,&baker_lock);
			/* debug("Woke up");*/
		}

		/* Reduce the bakers available and unlock socket */
		num_bakers--;
		pthread_mutex_unlock(&baker_lock);

		/* Throw a baker thread with pizzatype as argument */
		pthread_create(&bakers[c], &attr, baker_thread, &order->pizzas[c]);

	}

	/* Joining the bakers */
	for(i=num_pizzas-1;i>=0;i--){
		pthread_join(bakers[i], NULL);
		order->status--;
	}

	debug("Finished baking");

	/* ------------------------------------------------------- DELIVERY THREAD ---------------
	 * --------------------------------------------------------------------------------------*/
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	/* Locking the delivery_lock to check the num_deliveries value and decrement it */
	pthread_mutex_lock(&delivery_lock);

	/* waiting for condition variables with 'while' instead of 'if' is encouraged due to spurious wakeups */
	while(num_deliveries<=0){
		/* same apply as the cond_wait for the bakers */
		pthread_cond_wait(&free_deliveries,&delivery_lock);
	}

	num_deliveries--;
	pthread_mutex_unlock(&delivery_lock);

	/* Throw a delivery thread with distance type as argument */
	pthread_create(&delivery, &attr, delivery_thread, &order->type);

	pthread_attr_destroy(&attr);

	/* Wait/Join the delivery thread*/
	pthread_join(delivery, NULL);

	debug("Finished delivering");

	/*  After the delivering is finished, we must cancel the delay thread that
	 *  counts to TVERYLONG */
	pthread_cancel(delay);

	/* Delete the order from the list */
	delete(order);

	write(socket,"DONE!\0",6);
	close(socket);

	pthread_exit(NULL);
}

/* The baker thread */
void *baker_thread(void *arg){
	int index;

	/* Each dynamically created baker creates its own condition variable and mutex */
	pthread_mutex_t mut;
	pthread_cond_t cond;
	pizzaType pizzat = *(pizzaType *) arg;
	struct timespec time_to_wait = {0,0};

	/* The type of the pizza, get its time from the getPizzaTime table from common.h */
	time_to_wait.tv_sec = time(NULL) + getPizzaTime[pizzat];

	pthread_mutex_init(&mut,NULL);
	pthread_cond_init(&cond,NULL);

	pthread_mutex_lock(&mut);
	pthread_cond_timedwait(&cond, &mut, &time_to_wait);
	pthread_mutex_unlock(&mut);

	pthread_mutex_destroy(&mut);
	pthread_cond_destroy(&cond);

	/* debug("done baking"); */

	/* Protection of the num_bakers var */
	pthread_mutex_lock(&baker_lock);
	num_bakers++;
	pthread_mutex_unlock(&baker_lock);

	/* After finished baking, signal the free_bakers so other orders
	 * know that there are free_bakers now */
	pthread_cond_signal(&free_bakers);

	pthread_exit(NULL);
}

/* The delivery thread */
void *delivery_thread(void *arg){
	int index;

	/* Each dynamically created delivery creates its own condition variable and mutex */
	pthread_mutex_t mut;
	pthread_cond_t cond;
	struct timespec time_to_wait = {0,0};
	distanceType dist = *(distanceType* ) arg;

	/* The distance type, get its time from the getDistanceTime from common.h */
	time_to_wait.tv_sec = time(NULL) + getDistanceTime[dist];

	pthread_mutex_init(&mut,NULL);
	pthread_cond_init(&cond,NULL);

	pthread_mutex_lock(&mut);
	pthread_cond_timedwait(&cond, &mut, &time_to_wait);
	pthread_mutex_unlock(&mut);

	pthread_mutex_destroy(&mut);
	pthread_cond_destroy(&cond);

	/*  debug("done delivering"); */

	/* Protecting num_deliveries */
	pthread_mutex_lock(&delivery_lock);
	num_deliveries++;
	pthread_mutex_unlock(&delivery_lock);

	/* After finished delivering, signal the free_deliveries so other orders
	 * know that there are free deliveries now */
	pthread_cond_signal(&free_deliveries);

	pthread_exit(NULL);

}

/* The delayed thread */
void *delayed_thread(void *arg){
	struct timespec time_to_wait = {0,0};
	int socket = *(int*)arg;
	int j;

	pthread_mutex_t mut;
	pthread_cond_t cond;

	pthread_mutex_init(&mut,NULL);
	pthread_cond_init(&cond,NULL);

	/* This loop sleep(tverylong) and then sends coca cola
	 * to its corresponding client until cancelled by the order thread */
	for(j=0;;j++){
		time_to_wait.tv_sec = time(NULL) + TVERYLONG;

		pthread_mutex_lock(&mut);
		/* waiting for condition variables with 'while' instead of 'if' is encouraged due to spurious wakeups */
		while(pthread_cond_timedwait(&cond, &mut, &time_to_wait)!=ETIMEDOUT){
		}
		pthread_mutex_unlock(&mut);

		/* sleep(TVERYLONG); */
		write(socket,"Sorry for the delay you will receive a free cocacola\0",53);
	}
}

/* ------------------------------------------------------   STRUCTURE FUNCTIONS-------------
 * ========================================================================================
 * ---------------------------------------------------------------------------------------*/

/* ------------------------------     LIST    ----------------------------------------- */

/* Tries to insert an order in the list */
order_info *insert(pthread_t thread){

	int offset;
	order_info *ret;

	/* protected acces to the list.offset */
	pthread_mutex_lock(&list.lock);

	/* insert order thread in the list */
	offset = list.offset;

	pthread_mutex_unlock(&list.lock);

	/* if list is full then the list.offset var has no more used
	 * and new orders take its position in the table by pulling free
	 * positions from a stack */
	if (offset >= MAX_ORDERS){
		debug("pulling");
		offset = pull();
		/*  printstack(); */
	}else
		list.offset++;

	pthread_mutex_lock(&list.lock);

	/* Insertion in the double linked list */
	if (list.head == -1){		 /*  if list is empty */
		list.head = offset;
		list.order[offset].prev = -1;
	}else{
		list.order[list.tail].next = offset;
		list.order[offset].prev = list.tail;
	}
	list.tail = offset;
	list.order[offset].next = -1;

	/* at var shows the order position and thread its order_thread */
	list.order[offset].at = offset;
	list.order[offset].thread = thread;

	/* save the returned value before unlocking the mutex
	 * in order to avoid possible race conditions */
	ret = &list.order[offset];

	pthread_mutex_unlock(&list.lock);

	return ret;
}

/* Deletes an order from the list and push the freed position in the stack */
void delete(order_info *addr){

	int offset = addr->at;

	pthread_mutex_lock(&list.lock);

	/* Deletion from the double linked list */
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

	/* Pushes the freed position in the stack of free positions */
	push(offset);
	/* printstack();*/

	pthread_mutex_unlock(&list.lock);
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


/* ------------------------------      STACK    ----------------------------------------- */

/* Pushes in the stack the given free position(index of the list.order table) */
void push(int a){

	/* The stack is also mutex protected */
	pthread_mutex_lock(&opt.lock);

	/* Unlike pull, the push function don't need to check if stack is full
	 * because the way the orders are handled is at chunks of MAX_ORDERS
	 * so there are always free positions for pushing in the stack */

	opt.top++;
	opt.stack[opt.top] = a;

	pthread_mutex_unlock(&opt.lock);

	/* When a position has been pushed in the stack, the opt.full is signaled
	 * to wake (one at a time) orders that are waiting in pull because there are no free positions
	 * at the moment */
	pthread_cond_signal(&opt.full);
}

/* Pull the first free position from the stack (top) */
int pull(void){
	int ret;

	pthread_mutex_lock(&opt.lock);

	/* If all the list is occupied, there are no free positions in the stack so the order is waiting
	 * for opt.lock
	 *
	 * waiting for condition variables with 'while' instead of 'if' is encouraged due to spurious wakeups */
	while(opt.top  <= -1){
		debug("waiting");
		pthread_cond_wait(&opt.full,&opt.lock);
	}

	/* The returned value is saved before unlocking to avoid race conditions */
	ret = opt.stack[opt.top];
	opt.top--;

	pthread_mutex_unlock(&opt.lock);

	return ret;
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
