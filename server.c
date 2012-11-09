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
#include <sys/wait.h>		/* Declarations for waiting */
#include <sys/ipc.h>		/* XSI interprocess communication access structure */
#include <sys/shm.h>		/* XSI shared memory facility */
#include <semaphore.h>		/* semaphores */
#include <fcntl.h>			/* file control options */
//#include <sys/stat.h>
#include <signal.h>			/* signals */
#include <sys/time.h> 		/* time control */
//#include <time.h>

/* shared memory identifier */
#define SHMGLOBAL 5070

/* semaphore identifiers */
#define SEM_ORD_PEND "pending"
#define SEM_ORD_READY "done"
#define SEM_BAKERS "bakers"
#define SEM_DELIVERIES "deliveries"

/* ----------------------------------------------------------------------------------------
 * the order information is its client socket file descriptor and its status
 * also the order_info struct is used as an object in a list in shared memory
 * I use a double linked list for faster times of insertion and deletion
 * also it holds the pid of the status so that it can identify the socket to send the coca cola
 * ---------------------------------------------------------------------------------------*/
typedef struct _order{
	struct _order *prev;
	struct _order *next;
	int client_soc;
	unsigned int status;
	pid_t pid;
}order_info;

/* -----------------------------------------------------------------------------------------
 *  4 pointers to handle a list in shared memory
 *	START  shows the address taken from shmget
 *	head   shows the address of the head of the list
 *	end    shows the address of the end of the list
 *	offset shows the relative position of the next free segment
 *
 *  top  shows the top of the stack for memory management of this list
 *
 *	stack_start shows the starting memory of the circular buffer that implements the queue
 *	stack_end   shows the ending memory of the circular buffer that implements the queue
 *	
 *	Deleting a node from the list pushes its address in a memory management queue
 *	When inserting a node, first chooses a free space from that queue.
 *	If this queue is empty the offset value shows the next free segment in the shared memory
 *-------------------------------------------------------------------------------------------
 */

/* struct node of queue used for better memory management */
typedef struct queue{
	int data;
	struct queue *next;
}node;

typedef struct _list{
	order_info *START;
	order_info *head;
	order_info *end;
	int offset;
	sem_t *sem;
	
	//queue management for circular buffer
	node *top;
	node *stack_start;
	node *stack_end;
}list_info;

/* Deletes a order_info node from shared memory
 * (the info of the shared memory is passed with list_info)*/
void deleteshm(order_info*,list_info*);

/* Creates an order_info, inserts it in the shared memory 
 * and returns a pointer to the shared memory */
order_info *insertshm(int , unsigned int ,pid_t, list_info*);

/* Sends a coca cola to the socket that matches the give pid */
void sendcola(pid_t);

/* global declaration of file descriptors and addresses of shared memories to use with sig_int() */
int fd1,fd2;
void *addr1,*addr2;

/* Creation of 2 lists for the pending pizzas and the ready pizzas */
list_info *pending;
list_info *ready;

/* Declaration of 4 semaphores, 2 for the shared memory segments and 2 for the bakers and the deliveries */
sem_t *sem1,*sem2,*sem_bak,*sem_del;

/* Global declaration of socket file descriptor to use with sig_int() */
int listenfd;

/* ======================================================================================
 *                                                                       SIGNAL HANDLERS
 * ======================================================================================*/
/*  Function to properly release resources when a process is terminated with Ctrl-C */
void sig_int (int sig){
	close(listenfd);

	/* detach and delete the 2 shm */
	shmdt(addr1);
	shmctl(fd1,IPC_RMID,NULL);
	shmdt(addr2);
	shmctl(fd2,IPC_RMID,NULL);

	/*  unlink and close the 4 sem */
	sem_close(sem1);
	sem_close(sem2);
	sem_close(sem_bak);
	sem_close(sem_del);
	sem_unlink(SEM_ORD_PEND);
	sem_unlink(SEM_ORD_READY);
	sem_unlink(SEM_BAKERS);
	sem_unlink(SEM_DELIVERIES);

	exit(0);
}

/*  Helper function to avoid zombie processes */
void sig_chld( int sig) {
       pid_t pid;
       int stat;

       while ( ( pid = waitpid( -1, &stat, WNOHANG ) ) > 0 ) {
              //printf( "%s------------Child %d terminated.%s\n",KRED,pid,KNRM );
        }
 }

/* Invoked by the timer in each of the order childs */
void sig_alarm(int sig){
	sendcola(getpid());
}
/* ======================================================================================
 *                                                                                 MAIN 
 * ======================================================================================
 */
pid_t parent;

void showsem(char * message,sem_t *sem){
	int val;
	sem_getvalue(sem,&val);
	printf("%s[SEMAPHORE] - %d - %s%s\n",KYEL,val,message,KNRM);
}

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
	sa.sa_flags=0;
	sa.sa_handler=sig_chld;
	sigemptyset(&sa.sa_mask);
	if (sigaction(SIGCHLD,&sa,NULL) == -1)
			fatal("sigaction child");

	/* instead of closing correctly terminates and takes care of resources */
	sa2.sa_flags=0;
	sa2.sa_handler=sig_int;
	sigemptyset(&sa2.sa_mask);
	if (sigaction(SIGINT,&sa2,NULL) == -1)
		fatal("sigaction for Ctrl-C");

	/* ==================================================================================
	 *                                                               SHARED MEMORY SETUP
	 * - shared memory for the list of pending orders 
	 * - shared memory for the list of ready orders 
	 * ==================================================================================
	 */
	key_t key1 = SHMGLOBAL;
	key_t key2 = SHMGLOBAL+1;

	size_t shm_size1 = MAX_ORDERS*sizeof(order_info)+sizeof(list_info)+LISTENQ*sizeof(node);
	size_t shm_size2 = MAX_ORDERS*sizeof(order_info)+sizeof(list_info)+LISTENQ*sizeof(node);

	if ((fd1 = shmget(key1,shm_size1,IPC_CREAT | 0666))==-1)
		fatal("in shmget 1");
	if ((fd2 = shmget(key2,shm_size2,IPC_CREAT | 0666))==-1)
		fatal("in shmget 2");
	
	if ((addr1 = shmat(fd1,NULL,0)) == (void*)-1)
		fatal("in shmat 1");
	if ((addr2 = shmat(fd2,NULL,0)) == (void*)-1)
		fatal("in shmat 2");

	/* the pending list holds the starting point of shared memory3,
	 * the head and the and of the list and the offset that shows the next free space
	 */
	pending = addr1 + MAX_ORDERS * sizeof(order_info);
	pending->stack_start = addr1 + MAX_ORDERS * sizeof(order_info) + sizeof(list_info);
	pending->stack_end = addr1 + shm_size1; 

	pending->START = addr1;
	pending->head = 0;
	pending->end = 0;
	pending->top = 0;
	pending->offset = 1;

	/* the ready list holds the starting point of shared memory4,
	 * the head and the and of the list and the offset that shows the next free space
	 */
	ready = addr2 + MAX_ORDERS * sizeof(order_info);
	ready->stack_start = addr2 + MAX_ORDERS * sizeof(order_info) + sizeof(list_info);
	ready->stack_end = addr2 + shm_size2; 

	ready->START = addr2;
	ready->head = 0;
	ready->end = 0;
	ready->top = 0;
	ready->offset = 1;


	/* ==================================================================================
	 *                                                                 SEMAPHORES SETUP
	 * - sem1 for the shared_memory that holds the pending list
	 * - sem2 for the shared_memory that holds the ready list for delivery
	 * - sem_bak for the bakers status 
	 * - sem_del for the deliveries status
	 * ==================================================================================
	 */
	sem_unlink(SEM_ORD_PEND);
	sem_unlink(SEM_ORD_READY);
	sem_unlink(SEM_BAKERS);
	sem_unlink(SEM_DELIVERIES);

	sem1 = sem_open(SEM_ORD_PEND,O_CREAT|O_RDWR,S_IRUSR|S_IWUSR, 1);
	if (sem1 == SEM_FAILED)
		fatal("in sem_open 1");
	sem2 = sem_open(SEM_ORD_READY,O_CREAT|O_RDWR,S_IRUSR|S_IWUSR, 1);
	if (sem2 == SEM_FAILED)
		fatal("in sem_open 2");
	sem_bak = sem_open(SEM_BAKERS,O_CREAT|O_RDWR,S_IRUSR|S_IWUSR, NBAKERS);
	if (sem_bak == SEM_FAILED)
		fatal("in sem_open bak");
	sem_del = sem_open(SEM_DELIVERIES,O_CREAT|O_RDWR,S_IRUSR|S_IWUSR, NDELIVERY);
	if (sem_del == SEM_FAILED)
		fatal("in sem_open del");

	/* each list (pending/ready) contains 
	 * a pointer to its shared memory and to its semaphore */
	pending->sem = sem1;
	ready->sem = sem2;

	/*===================================================================================
	 *                                                                    SOCKETS SETUP
	 *===================================================================================
	 */
	int sockfd;
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
			if (errno == EINTR ) /* If interrupt received the system call continues normally*/
				continue;
			else
				fatal("in accepting connection");
		}

		/* fork to handle an order exclusively */
		childpid=fork();

		/* ------------------------------------------------------------------------------
		 *                                                                CHILD PROCESS
		 * -----------------------------------------------------------------------------*/
		if(childpid==0){ /* child process */
			close(listenfd);/* no reason to continue listening for orders */
			
			/* sigalarm handler */
			struct sigaction sig;
			sig.sa_flags=0;
			sig.sa_handler=sig_alarm;
			sigemptyset(&sig.sa_mask);
			if (sigaction(SIGALRM,&sig,NULL) == -1)
				fatal("sigaction child");

			//setting vars for use with setitimer TVERYLONG
			struct itimerval itv;
			itv.it_value.tv_sec = TVERYLONG/1000;			/*  it_value holds the time to sleep */
			itv.it_value.tv_usec = TVERYLONG%1000 * 1000;
			itv.it_interval.tv_sec = 0;			/*  it_interval holds the time to  */
			itv.it_interval.tv_usec = 0;

			//setting vars for use with nanosleep
			struct timespec request,remain;

			int recv_len = (NPIZZAS+2)*sizeof(char);
			write(sockfd, "Server: Pizza Ceid, tell me your order!\n",39);
			if (read(sockfd,&buffer,recv_len)==0){
				printf ("Rude client hanged up\n");
				exit(0);
			}
				
			printf("%s============== Received order %d, the order %s ================%s\n",KGRN,getpid(),buffer,KNRM);

			/* Starts the timer */
			if (setitimer(ITIMER_REAL,&itv,0) == -1)
				fatal("in setitimer");
				
			/* converts the buffer to int codes */
			int i = 0;
			pizzaType pizzas[NPIZZAS];
			for(i;i<strlen(buffer)-1;i++){
				pizzas[i] = buffer[i]-'0';
			}

			/* ==========================================================================
			 *                                                                 BAKING
			 * ==========================================================================
			 */

			/* order parser */
			int num_pizzas=strlen(buffer)-1;

			/* Insert the order in the shared memory shm3 for pending deliveries */
			order_info *addr = insertshm(sockfd,num_pizzas,getpid(),pending);
			//debug("insertshm",getpid());

			int c=num_pizzas;
			
			/* Begin loop for pizzas to bake */
			while(c>0){
				sem_wait(sem_bak);
				int bakerpid = fork();

				/* ------------------------------------------------------------------
				 *                                                     BAKER PROCESS
				 * ------------------------------------------------------------------
				 */
				if (bakerpid==0){ /* baker process */
					/* Sleeping */
					request.tv_sec = (long)getPizzaTime[pizzas[c-1]]/1000;
					request.tv_nsec = (long)getPizzaTime[pizzas[c-1]]%1000*1000000;
					int s = nanosleep(&request,&remain);
					if (s == -1 && errno !=EINTR)
						fatal("in nanosleep");

					sem_wait(sem1);
					addr->status-=1;
					sem_post(sem1);

					sem_post(sem_bak);

					sem_close(sem1);
					sem_close(sem2);
					sem_close(sem_bak);
					sem_close(sem_del);

					_exit(0);
				}//END OF BAKER PROCESS

				/* i reduce the c before the pizza is ready(addr->status)
				 * to show that i gave the order to a baker */
				c--;
			}//the loop will end when the order process has succesfully given all of its pizzas to bakers
			
			//and now it constantly checks its shared memory for ready status
			int baked=0;
			while(!baked){
				int check = (addr->status==0)?1:0;
				if (check){
					//printf("DONE baking\n");
					baked=1;
				}
			}

			//delete from shared memory shm1 
			sem_wait(sem1);
			deleteshm(addr,pending);
			//debug("deleteshm",getpid());
			sem_post(sem1);

		 	/* ==========================================================================
			 *                                                                 DELIVERY
			 * ==========================================================================
			 */
			
			/* the type is 0 if near and 1 if far according to the enum in common.h */
			distanceType type = buffer[strlen(buffer)-1]-'0';

			//insert in shared memory shm2
			addr = insertshm(sockfd,type,0,ready);//when the type becomes 2 the delivery is done
			//debug("insertshm",getpid());

			c = 1;
			while(c){
				sem_wait(sem_del);
				int delpid = fork();
					
				/* ==================================================================
				 *                                                  DELIVERY PROCESS
				 * ==================================================================
				 */
				if (delpid==0){ /*  delivery process */
					/* Sleeping */
					request.tv_sec = (long)getDistanceTime[type]/1000;
					request.tv_nsec = (long)getDistanceTime[type]%1000*1000000;
					int s = nanosleep(&request,&remain);
					if (s == -1 && errno !=EINTR)
						fatal("in nanosleep");

					sem_wait(sem2);
					addr->status=2;
					sem_post(sem2);
						
					sem_post(sem_del);

					sem_close(sem1);
					sem_close(sem2);
					sem_close(sem_bak);
					sem_close(sem_del);

					_exit(0);							
				}//END OF DELIVERY PROCESS

				/* i reduce c before pizza is delivered, to show that i gave the order to
				 * a delivery boy */
				c--;
			}//the loop will end when the order process has succesfully given its pizzas to

			//a delivery and now it constantly checks its shared memory for ready status
			int done=0;
			while(!done){
				int check = (addr->status==2)?1:0;
				if(check){
					//printf("DONE delivering\n");
					done=1;
				}
			}

			//delete from shared memory shm2
			sem_wait(sem2);
			deleteshm(addr,ready);
			//debug("deleteshm",getpid());
			sem_post(sem2);

			sem_close(sem1);
			sem_close(sem2);
			sem_close(sem_bak);
			sem_close(sem_del);
			write(sockfd,"DONE!",5);
			close(sockfd);

			_exit(0);
		}//END OF THE ORDER PROCESS

		close(sockfd);
	}
}

/* ======================================================================================
 * 																	       MEMORY STACK
 * Functions to manage the stack of free memory space
 *	
 *		stack_start              top                stack_end
 *		    |##|    <-  |##|  <- |##| .................
 * ======================================================================================
 */
void printstack(list_info *list){
	if (list->sem==sem1)printf("---Stack of pending is:TOP--> ");
	if (list->sem==sem2)printf("---Stack of ready is:TOP--> ");
	node *next = list->top;
	while(next!=0){
		printf("%d ",next->data);
		next = next->next;
	}	
	printf("<--BOTTOM\n");
}

void push(int s,list_info *list){
	node *new;
	if (list->top == 0) // stack is empty
		new = list->stack_start;
	else
		new = list->top + 1;
	
	if(new==list->stack_end){
		printf("%s[DEBUG] - out of free stack while pushing %d %s\n",KMAG,s,KNRM);
		printstack(list);
		kill(parent,SIGINT);
	}
	else{
		new->data=s;
		new->next=list->top;
		list->top=new;
	}
	printstack(list);
}

int pull(list_info *list){
	int ret;
	if (list->top == 0)
		ret = 0;
	else{
		ret=list->top->data;
		list->top=list->top->next;
	}
	return ret;
} 


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
	printf("---List: ");
	while (next!=0){
		printf("%d ",next->pid);
		next=next->next;
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
order_info *insertshm(int sockfd,unsigned int status,pid_t pid, list_info *list){
	/* -------------------------------------------------------------------------
	 * dynamic choice of free shared memory 
	 * if a node gets deleted its address is pushed in a queue 
	 * the next time i write in shared memory i choose first a free address
	 * from the queue, if null i choose the next segment pointed by the offset
	 * ------------------------------------------------------------------------*/
	int place=list->offset;
	int havemem=0;
	/* The first MAX_ORDERS orders will be placed in continuous positions in shared memory
	 * after these orders the next orders will take its position by pulling from a stack from 
	 * free position. The stack of free positions is populated after an order gets deleted.
	 * 
	 * eg. for a MAX_ORDERS of 50, the shared memory will have 50 positions for the data of each order
	 * when one of these first 50 gets deleted, it pushes its relative position into the stack so that 
	 * the order 50+1 can be saved in the list by pulling its address.
	 */
	if(list->offset>MAX_ORDERS){ 
		if(list->sem==sem1)debug("Throttled in pending",getpid());
		if(list->sem==sem2)debug("Throttled in ready",getpid());
		//write(4,"out of free memory",20);
		//_exit(0);
		do{ 
			place = pull(list);
			havemem = (place == 0)?0:1;
		}while (!havemem);
		printf("%d displacement: %d\n",getpid(),place);
		printstack(list);
	}else
		(list->offset)++;
	
	sem_wait(list->sem);
	if(list->sem==sem1)debug("insertshm in pending",getpid());
	if(list->sem==sem2)debug("insertshm in ready",getpid());
	order_info *addr = list->START + (place-1);

	addr->client_soc = sockfd;
	addr->status = status;
	addr->prev = 0;
	if (pid!=0)addr->pid = pid;
	if (list->head==0){//if list is empty
		list->head=addr;
	}else{
		list->end->next=addr;
		addr->prev=list->end;
	}
	list->end=addr;
	list->end->next = 0;
	
	//printlist(list);

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
	
	if(list->sem==sem1)debug("deleteshm in pending",getpid());
	if(list->sem==sem2)debug("deleteshm in ready",getpid());
	
	if (addr->prev != 0)//it is not the head
		addr->prev->next=addr->next;
	else{				//it is the head
		list->head=addr->next;
		if (list->head==0) //it is also the end
			list->end=0;
		else
			list->head->prev=0;
	}
	bzero(addr,sizeof(order_info));
	
	//printlist(list);
	
	/*  pushes the offset of the node in the queue, to show it is free from now on*/
	//printf("addr: %x , list->START: %x , difference %d\n",addr,list->START,addr-list->START);
	int offset = addr-list->START+1;
	push(offset,list);
	
	return;
}

/* --------------------------------------------------------------------------------------
 *																			SENDCOLA----
 * this function searches in the pending list for the given pid, it finds the socket of 
 * the client and sends him a message for a coca cola.
 * This is triggered by the signal handler individually by each order
 *
 * --------------------------------------------------------------------------------------
 */

void sendcola(pid_t pid){
	sem_wait(sem1);
	order_info *next = pending->head;
	sem_post(sem1);
	int found = 0;
	while(!found && next!=NULL){
		if (next->pid==pid){
			write(next->client_soc,"Sorry for the delay, you will receive a free coca cola\n",60);
			found=1;
		}
		next=next->next;
	}
}
