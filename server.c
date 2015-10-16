/*
 * =====================================================================================
 *
 *       Filename:  server.c
 *		  Project:  Operating Systems I - Project 1 - 5th Semester 2012
 *    Description:  Accepts orders.
 *
 *        Version:  1.2
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
#include <sys/wait.h>		/* Declarations for waiting */
#include <semaphore.h>		/* semaphores */
#include <fcntl.h>			/* file control options */
#include <signal.h>			/* signals */
#include <time.h>			/* for the POSIX timer functions */

#include <sys/mman.h>		/* for shared mapping mmap */

/* ----------------------------------------------------------------------------------------
 * the order information is its status and his process id
 * also the order_info struct is used as an object in a list in shared memory
 * I use a double linked list for faster times of insertion and deletion
 * ---------------------------------------------------------------------------------------*/
typedef struct _order{
	struct _order *prev;
	struct _order *next;
	unsigned int status;
	pid_t pid;
}order_info;

/* -----------------------------------------------------------------------------------------
 *  This struct holds information about:
 *  - the list in shared memory
 *		START  		shows the starting address of the list in shared memory
 *		head   		shows the address of the head of the list
 *		end    		shows the address of the end of the list
 *		offset 		shows the relative position of the next free segment
 *
 *	- the semaphores to handle the list
 *		sem	   		semaphore to lock the access in the list
 *		sem_res		for the pending list controls the bakers
 *					for the ready list controls the delivery boys
 *
 *  - the dynamic memory management information of the list
 *		stack_full	semaphore that shows how many places are full in the stack 
 *  	stack_empty	semaphore that shows how many places are empty in the stack
 *
 *  	top  		shows the top of the stack
 *		stack_start shows the starting memory of the stack
 *		stack_end   shows the ending memory of the stack
 *	
 *	Deleting a order_info from the list pushes a node with its address in a memory management stack
 *	that is controlled by 2 semaphores.When inserting an order_info in the list that exceeds the limits
 *	of the list, it is placed in unused memory segment that is pulled from the stack. 
 *-------------------------------------------------------------------------------------------
 */
#ifdef _STACKOP_
/* struct node of memory management stack */
typedef struct queue{
	int data;
	struct queue *next;
}node;
#endif

typedef struct _list{
	order_info *START;
	order_info *head;
	order_info *end;
	int offset;
	sem_t *sem,*sem_res;

#ifdef _STACKOP_
	/* stack management */
	sem_t *stack_full, *stack_empty;
	
	node *top;
	node *stack_start;
	node *stack_end;
#endif
}list_info;

/* Deletes a order_info node from shared memory
 * (the info of the shared memory is passed with list_info)*/
void deleteshm(order_info*,list_info*);

/* Creates an order_info, inserts it in the shared memory 
 * and returns a pointer to the shared memory */
order_info *insertshm(unsigned int ,pid_t, list_info*);

/* Given a pointer to list_info creates its semaphores returns -1 if it fails */
int init_sem(list_info *,unsigned int);

/* Pointers to 2 lists for the pending pizzas(for baking) and the ready pizzas(for delivery) */
list_info *pending;
list_info *ready;

#ifdef _DEBUG_
/*  debugging function to show info about a semaphore */
void showsem(char * message,sem_t *sem){
	int val;
	sem_getvalue(sem,&val);
	printf("%s[SEMAPHORE] - %d - %s%s\n",KYEL,val,message,KNRM);
}
#endif
/* ======================================================================================
 *                                                                       SIGNAL HANDLERS
 * ======================================================================================
 */
/*  Helper function to avoid zombie processes */
static void sig_chld( int sig) {
    pid_t pid;
    int stat;

    while ( ( pid = waitpid( -1, &stat, WNOHANG ) ) > 0 ) {
#ifdef _DEBUG_
        printf( "%s------------Child %d terminated.%s\n",KRED,pid,KNRM );
#endif 
    }
 }

/* I declare the sockfd up here because the server closes it everytime after accepting
 * a connection and as a result all the orders have the same socket number in their 
 * process space. So the sockfd is the same and i can write with this fixed number */
int sockfd;

/* Invoked by the timer in each of the order childs */
static void sig_timer(int sig){
	write(sockfd,"Sorry for the delay you will receive a free cocacola\0",53);	/* async-signal-safe (the use of write) */
}

/* ======================================================================================
 *                                                                                 MAIN 
 * ======================================================================================
 */
pid_t parent;


int main(int argc, char **argv){

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

	/* sigchld handler to avoid zombie process generation*/
	struct sigaction sa,sa2;
	sa.sa_flags= SA_RESTART;	/* useful because of bug #16 when baker interrupts sem_wait of its parent */
	sa.sa_handler=sig_chld;
	sigemptyset(&sa.sa_mask);
	if (sigaction(SIGCHLD,&sa,NULL) == -1)
			fatal("sigaction child");

	/* ==================================================================================
	 *                                                               SHARED MEMORY SETUP
	 * ==================================================================================
	 */
	
	/* Definitions of sizes */
	size_t list_size = MAX_ORDERS * sizeof(order_info);
	size_t list_info_size = sizeof(list_info);
#ifdef _STACKOP_
	size_t stack_size = LISTENQ * sizeof(node);
#else
	size_t stack_size = 0;
#endif
	size_t sem_size = 4 * sizeof(sem_t);
	size_t shm_size = list_size + list_info_size + stack_size + sem_size;

	void *addr = mmap(NULL, shm_size*2, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
	if (addr == MAP_FAILED)
		return -1;
	
	/* Initialization of list_info structs for the pending and the ready list */
	pending = addr + list_size;			/* Where list_info pending exists in shared memory */
	pending->START = addr;
#ifdef _STACKOP_
	pending->stack_start = (void *)pending + list_info_size;
	pending->stack_end = (void *)pending->stack_start + stack_size;
	pending->sem = (void *)pending->stack_end;
	pending->stack_full = pending->sem + 1;
	pending->stack_empty = pending->stack_full + 1;
	pending->sem_res = pending->stack_empty + 1;
	pending->top = 0;
#else
	pending->sem = (void *)pending + list_info_size;
	pending->sem_res = pending->sem + 1;
#endif
	pending->head = 0;
	pending->end = 0;
	pending->offset = 1;

	addr += shm_size;

	ready = addr + list_size;			/* Where list_info ready exists in shared memory */
	ready->START = addr;
#ifdef _STACKOP_
	ready->stack_start = (void *)ready + list_info_size;
	ready->stack_end = (void *)ready->stack_start + stack_size;
	ready->sem = (void *)ready->stack_end;
	ready->stack_full = ready->sem + 1;
	ready->stack_empty = ready->stack_full + 1;
	ready->sem_res = ready->stack_empty + 1;
	ready->top = 0;
#else
	ready->sem = (void *)ready + list_info_size;
	ready->sem_res = ready->sem + 1;
#endif
	ready->head = 0;
	ready->end = 0;
	ready->offset = 1;
	
	/* ==================================================================================
	 *                                                                 SEMAPHORES SETUP
	 * ==================================================================================
	 */
	/* init_sem initialize all the semaphores used by the structure */
	if (init_sem(pending,NBAKERS) == -1)
		fatal("in setup_sem for pending");
	if (init_sem(ready,NDELIVERY) == -1)
		fatal("in setup_sem for ready");

	/*===================================================================================
	 *                                                                    SOCKETS SETUP
	 *===================================================================================
	 */
	int listenfd;
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
		parent=getpid();
		client_size = sizeof(client_addr);

		sockfd = accept(listenfd, (struct sockaddr*)&client_addr,&client_size);

		if(sockfd < 0 ){
			if (errno == EINTR ) 		/* If interrupt received the system call continues normally*/
				continue;
			else
				fatal("in accepting connection");
		}

		/* fork to handle an order exclusively */
		childpid=fork();

		/* ------------------------------------------------------------------------------
		 *                                                                CHILD PROCESS
		 * -----------------------------------------------------------------------------*/
		if(childpid==0){ 		/* child process */
			close(listenfd);	/* no reason to continue listening for orders */
			
			/* sigalarm handler */
			struct sigaction sig;
			sig.sa_flags = SA_RESTART;		/* to restart any blocking calls */
			sig.sa_handler = sig_timer;
			sigemptyset(&sig.sa_mask);
			if (sigaction(SIGALRM,&sig,NULL) == -1)
				fatal("sigaction child");

			/* Creation of POSIX timer that sends sigalarm*/
			struct itimerspec ts;							/* settings for the timer */
			ts.it_value.tv_sec = TVERYLONG/1000;			/* it_value specifies the value the timer will expire */
			ts.it_value.tv_nsec= TVERYLONG%1000 * 1000000;	
			ts.it_interval.tv_sec = ts.it_value.tv_sec;		/* it_interval specifies if it is a periodic timer */
			ts.it_interval.tv_nsec = ts.it_value.tv_nsec;

			struct sigevent sev;
			timer_t tid;						/* timer handle */

			sev.sigev_notify = SIGEV_SIGNAL;	/* Notify via signal */
			sev.sigev_signo = SIGALRM;			/* Notify using SIGALRM signal */

			if(timer_create(CLOCK_REALTIME, &sev, &tid) == -1)
				fatal("in timer_create");

			/* Declaration of vars for use with nanosleep */
			struct timespec request,remain;

			/* Server gets order from client */
			int recv_len = (NPIZZAS+2);
			write(sockfd, "Server: Pizza Ceid, tell me your order!",40);
			if (read(sockfd,&buffer,recv_len)==0){	 	/* If clients disconnects before sends order */
				printf ("Rude client hanged up\n");
				exit(0);
			}
			
			printf("%s============== Received order %d, the order %s ================%s\n",KGRN,getpid(),buffer,KNRM);

			/* Starts the timer */
			if(timer_settime(tid,0,&ts,NULL) == -1)
				fatal("in timer_settime");

			/* converts the buffer to int codes */
			int i = 0;
			pizzaType pizzas[NPIZZAS];
			for(i;i<strlen(buffer )-1;i++){
				pizzas[i] = buffer[i]-'0';
			}

			/* ==========================================================================
			 *                                                                 BAKING
			 * ==========================================================================
			 */

			int num_pizzas=strlen(buffer)-1;

			/* Insert the order in the shared memory shm3 for pending deliveries */
			order_info *addr = insertshm(num_pizzas,getpid(),pending);

			int c=num_pizzas;
			
			/* Begin loop for pizzas to bake */
			for (c=0;c<num_pizzas;c++){
				/* pending->sem_res  is initialized with the number of bakers 
				 * it is reduced and throw a baker process	(bug #16 fixed)*/
				sem_wait(pending->sem_res);
				/* ------------ ----------------------------------------------------------
				 *                                                       BAKER PROCESS
				 * ----------------------------------------------------------------------
				 */
				switch(fork())  {
					case -1:
						fatal ("in fork");
					case 0: 	/* Baker process */
						/* Sleeping */
						request.tv_sec = getPizzaTime[pizzas[c]]/1000L;
						request.tv_nsec = getPizzaTime[pizzas[c]]%1000*1000000L;
						int s = nanosleep(&request,&remain);
						if (s == -1 && errno !=EINTR)
							fatal("in nanosleep");

						/* Redu ction of status of order (until 0) 
						 * pending->sem is the mutex that locks and unlocks the access in the pending list */
						sem_wait(pending->sem);
						addr->status-=1;
						sem_post(pending->sem);

						/* Baker has finished show increases the pending->sem_res semaphore for the bakers */
						sem_post(pending->sem_res);
#ifdef _DEBUG_	
						int sval;
						sem_getvalue(pending->sem_res,&sval);
						printf("sval: %d\n",sval);
#endif
						_exit(0);	//END OF BAKER PROCESS
					default:
						break;
				}
			}//the loop will end when the order process has succesfully given all of its pizzas to bakers

			/* Waiting for the childs(bakers) to finish */
			while((childpid = wait(NULL)) != -1)
					continue;
			if (errno != ECHILD)
				fatal("in wait");

			//delete from shared memory shm1 with blocking and unblocking of the signal 
			deleteshm(addr,pending);

		 	/* ==========================================================================
			 *                                                                 DELIVERY
			 * ==========================================================================
			 */
			
			/* the type is 0 if near and 1 if far according to the enum in common.h */
			distanceType type = buffer[strlen(buffer)-1]-'0';

			//insert in shared memory shm2
			addr = insertshm(type,0,ready);//when the type becomes 2 the delivery is done

			sem_wait(ready->sem_res);
			/* --------------------------------------------------------------------------
			 *                                                		  DELIVERY PROCESS
			 * --------------------------------------------------------------------------
			 */
			switch(fork()) {
				case -1:
					fatal("in delivery fork");
				case 0: /*  delivery process */
					/* Sleeping */
					request.tv_sec = (long)getDistanceTime[type]/1000;
					request.tv_nsec = (long)getDistanceTime[type]%1000*1000000;
					int s = nanosleep(&request,&remain);
					if (s == -1 && errno !=EINTR)
					fatal("in nanosleep");

					sem_wait(ready->sem);
					addr->status=2;
					sem_post(ready->sem);
					
					sem_post(ready->sem_res);

					_exit(0);	//END OF DELIVERY PROCESS						
				default:
					break;
			}

			/* Waiting for the childs(deliveries) to finish */
			while((childpid = wait(NULL)) != -1)
					continue;
			if (errno != ECHILD)
				fatal("in wait");
			
			//delete from shared memory shm2
			deleteshm(addr,ready);

			write(sockfd,"DONE!\0",6);
			close(sockfd);

			_exit(0);
		}//END OF THE ORDER PROCESS

		close(sockfd);
	}
}

/* Initialization of unnamed semaphores of the structure list_info 
 * the resources option is given in case the NBAKERS are not the same as NDELIVERY*/
int init_sem(list_info *list,unsigned int resources){
	if(sem_init(list->sem,1,1) == -1)
		return -1;
	if(sem_init(list->sem_res,1,resources) == -1)
		return -1;
#ifdef _STACKOP_
	if(sem_init(list->stack_full,1,MAX_ORDERS) == -1)
		return -1;
	if(sem_init(list->stack_empty,1,MAX_ORDERS) == -1)
		return -1;
	
	/* zero the list->stack_full semaphore */
	int i;
	for (i=0;i<MAX_ORDERS;i++)
		sem_wait(list->stack_full);
#endif
}

/* ======================================================================================
 * 									MEMORY STACK
 * Functions to manage the stack of free memory space
 *	
 *		stack_start              top                stack_end
 *		    |##|    <-  |##|  <- |##| .................
 * ======================================================================================
 */
#ifdef _STACKOP_
void printstack(list_info *list){
	if (list->sem==pending->sem)printf("---Stack of pending is:TOP--> ");
	if (list->sem==ready->sem)printf("---Stack of ready is:TOP--> ");
	node *next = list->top;
	while(next!=0){
		printf("%d ",next->data);
		next = next->next;
	}	
	printf("<--BOTTOM\n");
}

void push(int s,list_info *list){
	node *new;
	sem_wait(list->stack_empty);
#ifdef _DEBUG_
	showsem("stack_empty after wait",list->stack_empty);
#endif
	if (list->top == 0) // stack is empty
		new = list->stack_start;
	else
		new = list->top + 1;
	
	new->data=s;
	new->next=list->top;
	list->top=new;
	sem_post(list->stack_full);
#ifdef _DEBUG_
	printstack(list);
#endif
}

int pull(list_info *list){
	int ret;
	sem_wait(list->stack_full);
	ret=list->top->data;
	list->top=list->top->next;
	sem_post(list->stack_empty);
	return ret;
} 

#endif

/* ======================================================================================
 *                                                                          ORDERS LIST
 * Functions for handling the list of orders in shared memory 3
 * - insert
 * - delete
 * - list of free shared memory
 * ======================================================================================
 */

void printlist(list_info *list){
	order_info *next = list->head;
	if(list->sem==pending->sem)printf("---List of pending: ");
	if(list->sem==ready->sem)printf("---List of ready: ");
	int c=0;
	while (next!=0){
		printf("%d ",next->pid);
		next=next->next;
		if (c++>MAX_ORDERS){
			next=0;
			printf("-------------error in end of list\n");
		}
	}
	printf("\n");	
}

/* --------------------------------------------------------------------------------------
 *                                                                             INSERT----
 * Args:-sockfd of order's client
 * 		-status of baked pizzas of the order_info
 * 		-flag to choose between shared memory/list of pending pizzas and ready pizzas
 *
 * 	Given the list_info, the sockfd of the order and its status adds a node of order
 * 	in the list and returns a pointer of it in the shared memory.
 * 	
 * 	NOTE: for pending list the status shows the pizzas to be ready
 * 	 	  for ready list the status shows the type of the delivery (far/near)
 * --------------------------------------------------------------------------------------		
 */
order_info *insertshm(unsigned int status,pid_t pid, list_info *list){
	/* -------------------------------------------------------------------------
	 * dynamic choice of free shared memory 
	 * if a node gets deleted its address is pushed in a queue 
	 * the next time i write in shared memory i choose first a free address
	 * from the queue, if null i choose the next segment pointed by the offset
	 * ------------------------------------------------------------------------*/
	int place=list->offset;
	/* The first MAX_ORDERS orders will be placed in continuous positions in shared memory
	 * after these orders the next orders will take its position by pulling from a stack from 
	 * free position. The stack of free positions is populated after an order gets deleted.
	 * 
	 * eg. for a MAX_ORDERS of 50, the shared memory will have 50 positions for the data of each order
	 * when one of these first 50 gets deleted, it pushes its relative position into the stack so that 
	 * the order 50+1 can be saved in the list by pulling its address.
	 */
	if(place>MAX_ORDERS){ 
#ifdef _DEBUG_
		if(list->sem==pending->sem)debug("Throttled in pending",getpid());
		if(list->sem==ready->sem)debug("Throttled in ready",getpid());
#endif
#ifdef _STACKOP_	
		place = pull(list);
	#ifdef _DEBUG_
		printstack(list);
	#endif
#else
		write(4,"out of free memory",20);
		_exit(0);
#endif
#ifdef _DEBUG_
		printf("%d displacement: %d\n",getpid(),place);
#endif
	}else
		(list->offset)++;
#ifdef _DEBUG_	
	if(list->sem==pending->sem)debug("insertshm in pending",getpid());
	if(list->sem==ready->sem)debug("insertshm in ready",getpid());
#endif
	sem_wait(list->sem);
	order_info *addr = list->START + (place-1);

	addr->status = status;
	addr->prev = 0;
	if (pid!=0)addr->pid = pid;
	if (list->head==0){		/* if list is empty */
		list->head=addr;
	}else{
		list->end->next=addr;
		addr->prev=list->end;
	}
	list->end=addr;
	list->end->next = 0;
#ifdef _DEBUG_
	printlist(list);
#endif
	sem_post(list->sem);
	return addr;
}

/* --------------------------------------------------------------------------------------
 *                                                                             DELETE----
 * Given a pointer to the order_info object shared memory, memset the memory to 0,
 * fixes the pointers, adds the shared_memory section to the list of free sections
 * --------------------------------------------------------------------------------------
 */
void deleteshm(order_info *addr,list_info *list){
#ifdef _DEBUG_
	if(list->sem==pending->sem)debug("deleteshm in pending",getpid());
	if(list->sem==ready->sem)debug("deleteshm in ready",getpid());
#endif
	sem_wait(list->sem);
	if (addr->prev != 0)	/* it is not the head */
		addr->prev->next=addr->next;
	else{					/* it is the head */
		list->head=addr->next;
		if (list->head==0) 	/* it is also the end */
			list->end=0;
		else
			list->head->prev=0;
	}
	bzero(addr,sizeof(order_info));
	
	sem_post(list->sem);
	
#ifdef _STACKOP_
	/*  pushes the offset of the node in the queue, to show it is free from now on */
	/*	printf("addr: %x , list->START: %x , difference %d\n",addr,list->START,addr-list->START); */
	int offset = addr-list->START+1;
	push(offset,list);
#endif
	return;
}

