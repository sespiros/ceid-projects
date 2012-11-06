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
#include <sys/ipc.h>
#include <sys/shm.h>
#include <semaphore.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <signal.h>
#include <sys/time.h>
#include <time.h>

/*Size of request queue*/
#define LISTENQ  20

/* shared memory identifier */
#define SHMGLOBAL 5070

/* semaphore identifiers */
#define SEM_BAKERS "bakers"
#define SEM_DELIVERIES "deliveries"
#define SEM_ORD_PEND "pending"
#define SEM_ORD_READY "done"

/* max orders issued for setting the size of the 3rd shared memory */
#define MAX_ORDERS 200

/* ----------------------------------------------------------------------------------------
 * the order information is its client socket file descriptor and its status
 * also the order_info struct is used as an object in a list in shared memory
 * the struct i use is a double linked list for faster times of insertion and deletion
 * ---------------------------------------------------------------------------------------*/
typedef struct _order{
	struct _order *prev;
	struct _order *next;
	int client_soc;
	unsigned int status;
}order_info;

/* -----------------------------------------------------------------------------------------
 * 4 pointers to handle a list in shared memory
 *	START  shows the address taken from shmget
 *	head   shows the address of the head of the list
 *	end    shows the address of the end of the list
 *	offset shows the relative position of the next free segment
 *	front  shows the front of the queue for memory management of this list
 *	rear   shows the rear of the queue for memory management of this list
 *
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
	node *front;
	node *rear;
}list_info;

/* Deletes a order_info node from shared memory
 * (the info of the shared memory is passed with list_info)*/
void deleteshm(order_info*,list_info*);

/* Creates an order_info, inserts it in the shared memory 
 * and returns a pointer to the shared memory */
order_info *insertshm(int , unsigned int ,list_info*);

/* global declaration of file descriptors and addresses of shared memories to use with sig_int */
int fd1,fd2,fd3,fd4;
void *addr1,*addr2,*addr3,*addr4;

/* Creation of 2 lists for the pending pizzas and the ready pizzas */
list_info pending;
list_info ready;

/* Declaration of 3 semaphores for each of the shared memory segments */
sem_t *sem1,*sem2,*sem3,*sem4;

/*  Global declaration of socket file descriptor to use with sig_int*/
int listenfd;

/* ======================================================================================
 *                                                                       SIGNAL HANDLERS
 * ======================================================================================*/
/*  Function to properly release resources when a process is terminated with Ctrl-C */
void sig_int (int sig){
	close(listenfd);

	/* detach and delete the 4 shm */
	shmdt(addr1);
	shmctl(fd1,IPC_RMID,NULL);
	shmdt(addr2);
	shmctl(fd2,IPC_RMID,NULL);
	shmdt(addr3);
	shmctl(fd3,IPC_RMID,NULL);
	shmdt(addr4);
	shmctl(fd4,IPC_RMID,NULL);

	/*  unlink and close the 4 sem */
	sem_close(sem1);
	sem_close(sem2);
	sem_close(sem3);
	sem_close(sem4);
	sem_unlink(SEM_BAKERS);
	sem_unlink(SEM_DELIVERIES);
	sem_unlink(SEM_ORD_PEND);
	sem_unlink(SEM_ORD_READY);

	exit(0);
}

/*  Helper function to avoid zombie processes */
void sig_chld( int sig) {
       pid_t pid;
       int stat;

       while ( ( pid = waitpid( -1, &stat, WNOHANG ) ) > 0 ) {
              printf( "Child %d terminated.\n", pid );
        }
 }

void sig_alarm(int sig){
	printf("Coca colla received %d\n",getpid());
}
/* ======================================================================================
 *                                                                                 MAIN 
 * ======================================================================================
 */
int main(int argc, char **argv){
	
	/* definitions of standard times (used better with the enums defined in common.h)*/	
	int timeofPizza[]={450,120,150};
	int timeofClient[]={50,100};

	/* ------------------------------------------------------------
	 * holds the client order information in the following way
	 *      |===============================================|
	 * 		|pizza #1 | pizza #2 | pizza #NPIZZA | near/far |
	 *      |===============================================|
	 *		to order 2 peperonis, 1 special for far is 1|1|2|1
	 *		according to the enum defined in common.h
	 * ----------------------------------------------------- ----*/
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
	 * - shared memory for the status of bakers
	 * - shared memory for the status of deliveries
	 * - shared memory for the order statuses
	 * ==================================================================================
	 */
	key_t key1 = SHMGLOBAL;
	key_t key2 = SHMGLOBAL+1;
	key_t key3 = SHMGLOBAL+2;
	key_t key4 = SHMGLOBAL+3;

	size_t shm_size1 = sizeof(unsigned int);
	size_t shm_size2 = sizeof(unsigned int);
	size_t shm_size3 = MAX_ORDERS*sizeof(order_info);
	size_t shm_size4 = MAX_ORDERS*sizeof(order_info);

	if ((fd1 = shmget(key1,shm_size1,IPC_CREAT | 0666))==-1)
		fatal("in shmget 1");
	if ((fd2 = shmget(key2,shm_size2,IPC_CREAT | 0666))==-1)
		fatal("in shmget 2");
	if ((fd3 = shmget(key3,shm_size3,IPC_CREAT | 0666))==-1)
		fatal("in shmget 3");
	if ((fd4 = shmget(key4,shm_size4,IPC_CREAT | 0666))==-1)
		fatal("in shmget 4");
	
	if ((addr1 = shmat(fd1,NULL,0)) == (void*)-1)
		fatal("in shmat 1");
	if ((addr2 = shmat(fd2,NULL,0)) == (void*)-1)
		fatal("in shmat 2");
	if ((addr3 = shmat(fd3,NULL,0)) == (void*)-1)
		fatal("in shmat 3");
	if ((addr4 = shmat(fd4,NULL,0)) == (void*)-1)
		fatal("in shmat 4");

	/* pointer to shared memory1 that holds the number of bakers currently active */
	unsigned int *bak_info = addr1;
	*bak_info = 0;

	/* pointer to shared memory2 that holds the number of deliveries currently active */
	unsigned int *deliver_info = addr2;
	*deliver_info = 0;

	/* the pending list holds the starting point of shared memory3,
	 * the head and the and of the list and the offset that shows the next free space
	 */
	pending.START = addr3;
	pending.head = 0;
	pending.end = 0;
	pending.front = 0;
	pending.rear = 0;
	pending.offset = 0;

	/* the ready list holds the starting point of shared memory4,
	 * the head and the and of the list and the offset that shows the next free space
	 */
	ready.START = addr4;
	ready.head = 0;
	ready.end = 0;
	ready.front = 0;
	ready.rear = 0;
	ready.offset = 0;

	/* =================================================================================
	 *                                                                 SEMAPHORES SETUP
	 * - sem1 for the bakers status shared memory
	 * - sem2 for the deliveries status shared memory
	 * - sem3 for the shared_memory that holds the pending list
	 * - sem4 for the shared_memory that holds the ready list for delivery
	 * =================================================================================
	 */
	sem_unlink(SEM_BAKERS);
	sem_unlink(SEM_DELIVERIES);
	sem_unlink(SEM_ORD_PEND);
	sem_unlink(SEM_ORD_READY);

	sem1 = sem_open(SEM_BAKERS,O_CREAT|O_RDWR,S_IRUSR|S_IWUSR, 1);
	if (sem1 == SEM_FAILED)
		fatal("in sem_open 1");
	sem2 = sem_open(SEM_DELIVERIES,O_CREAT|O_RDWR,S_IRUSR|S_IWUSR, 1);
	if (sem2 == SEM_FAILED)
		fatal("in sem_open 2");
	sem3 = sem_open(SEM_ORD_PEND,O_CREAT|O_RDWR,S_IRUSR|S_IWUSR, 1);
	if (sem3 == SEM_FAILED)
		fatal("in sem_open 3");
	sem4 = sem_open(SEM_ORD_READY,O_CREAT|O_RDWR,S_IRUSR|S_IWUSR, 1);
	if (sem4 == SEM_FAILED)
		fatal("in sem_open 4");

	/* each list (pending/ready) contains 
	 * a pointer to its shared memory and to its semaphore */
	pending.sem = sem3;
	ready.sem = sem4;

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

			//setting vars for use with setitimer
			struct itimerval itv;
			itv.it_value.tv_sec = 0;			/*  it_value holds the time to sleep */
			itv.it_value.tv_usec = TVERYLONG * 1000;
			itv.it_interval.tv_sec = 0;			/*  it_interval holds the time to  */
			itv.it_interval.tv_usec = 0;

			//setting vars for use with nanosleep
			struct timespec request,remain;
			request.tv_sec = 0L;

			int recv_len = (NPIZZAS+2)*sizeof(char);
			write(sockfd, "Server: Pizza Ceid, tell me your order!\n",39);
			recv(sockfd,&buffer,recv_len,0);
			printf("Received order from %d, %s\n",sockfd,buffer);

			/* Starts the timer */
			if (setitimer(ITIMER_REAL,&itv,0) == -1)
				fatal("in setitimer");
				
			/* converts the buffer to int codes */
			int i = 0;
			int codes[NPIZZAS];
			for(i;i<strlen(buffer);i++){
				codes[i] = buffer[i]-'0';
			}

			/* ==========================================================================
			 *                                                                 BAKING
			 * ==========================================================================
			 */

			/* order parser */
			int num_pizzas=strlen(buffer)-1;

			/* Insert the order in the shared memory shm3 for pending deliveries */
			order_info *addr = insertshm(sockfd,num_pizzas,&pending);

			int c=num_pizzas;
			
			/* Begin loop for pizzas to bake */
			while(c>0){
				sem_wait(sem1);
				if (*bak_info<NBAKERS){
					(*bak_info)++;
					sem_post(sem1);
					int bakerpid = fork();

					/* ------------------------------------------------------------------
					 *                                                     BAKER PROCESS
					 * ------------------------------------------------------------------
					 */
					if (bakerpid==0){ /* baker process */
						int pizzatype=codes[c-1];
						printf("pizza time: %d\n",timeofPizza[pizzatype]);
						request.tv_nsec = (long)timeofPizza[pizzatype]*1000000;
						//usleep(timeofPizza[pizzatype]*1000);
						int s = nanosleep(&request,&remain);
						if (s == -1 && errno !=EINTR)
							fatal("in nanosleep");

						sem_wait(sem3);
						addr->status-=1;
						//in sem block for correct debug	
						printf("I'm %d and i just baked pizza of type %d\n",getpid(),pizzatype);
						sem_post(sem3);

						sem_wait(sem1);
						(*bak_info)--;
						sem_post(sem1);

						sem_close(sem1);
						sem_close(sem2);
						sem_close(sem3);
						sem_close(sem4);

						exit(0);
					}//END OF BAKER PROCESS

					/* i reduce the c before the pizza is ready(addr->status)
					 * so that the next baker reads correct pizzatype
					 * doesn't affect the time of pizza because the actual state
					 * of the order is in shared memory (addr->status)
					 */
					c--;
				}else
					sem_post(sem1);
			}
			//the loop will end when the order process has succesfully given all of its pizzas to bakers
			//and now it constantly checks its shared memory for ready status
			int baked=0;
			while(!baked){
				sem_wait(sem3);
				int check = (addr->status==0)?1:0;
				sem_post(sem3);
				if (check){
					printf("DONE baking\n");
					baked=1;
				}
			}
			//delete from shared memory shm3 
			deleteshm(addr,&pending);

			/* ==========================================================================
			 *                                                                 DELIVERY
			 * ==========================================================================
			 */
			
			/* the type is 0 if near and 1 if far according to the enum in common.h */
			int type = codes[strlen(buffer)-1];

			//insert in shared memory shm4
			addr = insertshm(sockfd,type,&ready);//when the type becomes 2 the delivery is done

			c = 1;
			while(c){
				sem_wait(sem2);
				if (*deliver_info<NDELIVERY){
					(*deliver_info)++;
					sem_post(sem2);
					int delpid = fork();
					
					/* ==================================================================
					 *                                                  DELIVERY PROCESS
					 * ==================================================================
					 */
					if (delpid==0){ /*  delivery process */
						request.tv_nsec = (long)timeofClient[type]*1000000;
						//usleep(timeofClient[type]*1000);
						int s = nanosleep(&request,&remain);
						if (s == -1 && errno !=EINTR)
							fatal("in nanosleep");

						sem_wait(sem4);
						addr->status=2;
						printf("I'm %d and i just delivered order of type %d\n",getpid(),type);
						sem_post(sem4);
						
						sem_wait(sem2);
						(*deliver_info)--;
						sem_post(sem2);

						sem_close(sem1);
						sem_close(sem2);
						sem_close(sem3);
						sem_close(sem4);

						exit(0);							
					}//END OF DELIVERY PROCESS

					/* i reduce c before pizza is delivered, to show that i gave the order to
					 * a delivery boy */
					c--;
				}else
					sem_post(sem2);
			}
				
			//the loop will end when the order process has succesfully given its pizzas to
			//a delivery and now it constantly checks its shared memory for ready status
			int done=0;
			while(!done){
				sem_wait(sem4);
				int check = (addr->status==2)?1:0;
				sem_post(sem4);
				if(check){
					printf("DONE delivering\n");
					done=1;
				}
			}
			//delete from shared memory shm4
			deleteshm(addr,&ready);

			sem_close(sem1);
			sem_close(sem2);
			sem_close(sem3);
			sem_close(sem4);
			write(sockfd,"DONE!",5);
			close(sockfd);

			exit(0);
		}//END OF THE ORDER PROCESS

		close(sockfd);
	}
}


/* ======================================================================================
 * 																	       MEMORY QUEUE
 * Functions to manage the queue of free memory space
 *	
 *		front             rear
 *		 |##| <- |##|  <- |##|
 * ======================================================================================
 */
void printq(list_info *list){
	printf("---Queue of free memory is: ");
	node *next = list->rear;
	while(next!=0){
		printf("%d ",next->data);
		next = next->next;
	}
	printf("\n");
}

void insert_front(int s,list_info *list){
	node *new;
	new = (node *)malloc(sizeof(node));
	new->data = s;
	new->next = 0;
	if (list->front == 0)
		list->rear=new;
	else
		list->front->next=new;
	list->front=new;	
}

int remove_rear(list_info *list){
	int ret;
	if (list->rear!=0){
		node *temp=list->rear;
		list->rear = temp->next;
		ret=temp->data;
		free(temp);
	}else
		ret=0;
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
	order_info *next = list->START;
	printf("---List: ");
	while (next!=0){
		printf("%d ",next->client_soc);
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
order_info *insertshm(int sockfd,unsigned int status,list_info *list){
	/* -------------------------------------------------------------------------
	 * dynamic choice of free shared memory 
	 * if a node gets deleted its address is pushed in a queue 
	 * the next time i write in shared memory i choose first a free address
	 * from the queue, if null i choose the next segment pointed by the offset
	 * ------------------------------------------------------------------------*/
	int place;	
 	if((place = remove_rear(list))==0)
		place = (list->offset)++;		
	
	order_info *addr = list->START + place*sizeof(order_info);
	sem_wait(list->sem);
	addr->client_soc=sockfd;
	addr->status=status;
	sem_post(list->sem);
	if (list->head==0){//if list is empty
		list->head=addr;
		list->end=addr;
	}else{
		sem_wait(list->sem);
		list->end->next=addr;
		addr->prev=list->end;
		list->end=addr;
		sem_post(list->sem);
	}
	printlist(list);
	return addr;
}

/* --------------------------------------------------------------------------------------
 *                                                                             DELETE----
 * Given a pointer to the order_info object shared memory, memset the memory to 0,
 * fixes the pointers, adds the shared_memory section to the list of free sections
 * --------------------------------------------------------------------------------------
 */
void deleteshm(order_info *addr,list_info *list){
	
	/*  pushes the offset of the node in the queue, to show it is free from now on*/
	int offset = (addr-list->START)/sizeof(order_info);
	insert_front(offset,list);

	sem_wait(list->sem);
	if (addr->prev != 0)
		addr->prev->next=addr->next;
	bzero(addr,sizeof(order_info));
	sem_post(list->sem);
	printq(list);
	return;
}
