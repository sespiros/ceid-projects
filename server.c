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

/*Size of request queue*/
#define LISTENQ  20

/* shared memory identifier */
#define SHMGLOBAL 5070

/* semaphore identifiers */
#define SEM_BAKERS "bakers"
#define SEM_DELIVERIES "deliveries"
#define SEM_ORD_PEND "pending"
#define SEM_ORD_DONE "done"

/* max orders issued for setting the size of the 3rd shared memory */
#define MAX_ORDERS 200

/*  Helper function to avoid zombie processes */
void sig_chld(int signo);

/* the order information is its client socket file descriptor and its status */
typedef struct _order{
	struct _order *prev;
	struct _order *next;
	int client_soc;
	unsigned int status;
}order_info;

/*  4 pointers to handle a list in shared memory
 *	START  shows the address taken from shmget
 *	head   shows the address of the head of the list
 *	end    shows the address of the end of the list
 *	offset shows the relative position of the next free segment
 *	
 *	Deleting a node from the list pushes its address in a memory management queue
 *	When inserting a node, first chooses a free space from that queue.
 *	If this queue is empty the offset value shows the next free segment in the shared memory
 *
 *  */
typedef struct _list{
	order_info *START;
	order_info *head;
	order_info *end;
	int offset;
}list_info;

/* Declaration of list functions and pointers */
void delete(order_info*);
order_info *insert(int , unsigned int ,list_info*);

/* Creation of 2 lists for the pending pizzas and the ready pizzas */
list_info pending;

/* Declaration of 3 semaphores for each of the shared memory segments */
sem_t *sem1,*sem2,*sem3;

int main(int argc, char **argv){
	
	/* definitions of standard times (used better with the enums defined in common.h)*/	
	int timeofPizza[]={100,120,150};
	int timeofClient[]={50,100};

	/* holds the client order information in the following way
	 *      |-----------------------------------------------|
	 * 		|pizza #1 | pizza #2 | pizza #NPIZZA | near/far |
	 *      |-----------------------------------------------|
	 *		to order 2 peperonis, 1 special for far is 1|1|2|1
	 *		according to the enum defined in common.h
	 */
	char buffer[NPIZZAS+1];

	signal(SIGCHLD, sig_chld); 	/* sigchld handler to avoid zombie process generation*/

	/*-----------------------------------------------------------------------------------
	 * Standard socket creation 
	 *-----------------------------------------------------------------------------------
	 */
	int listenfd, sockfd;
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
	 * setting shared memory of the server
	 * - shared memory for the status of bakers
	 * - shared memory for the status of deliveries
	 * - shared memory for the order statuses
	 * ----------------------------------------------------------------------------------
	 */
	int fd1,fd2,fd3;
	key_t key1 = SHMGLOBAL;
	key_t key2 = SHMGLOBAL+1;
	key_t key3 = SHMGLOBAL+2;

	size_t shm_size1 = sizeof(unsigned int);
	size_t shm_size2 = sizeof(unsigned int);
	size_t shm_size3 = MAX_ORDERS*sizeof(order_info);

	if ((fd1 = shmget(key1,shm_size1,IPC_CREAT | 0666))==-1)
		fatal("in shmget 1");
	if ((fd2 = shmget(key2,shm_size2,IPC_CREAT | 0666))==-1)
		fatal("in shmget 2");
	if ((fd3 = shmget(key3,shm_size3,IPC_CREAT | 0666))==-1)
		fatal("in shmget 3");
	
	void *addr1;
	void *addr2;
	void *addr3;
	if ((addr1 = shmat(fd1,NULL,0)) == (void*)-1)
		fatal("in shmat 1");
	if ((addr2 = shmat(fd2,NULL,0)) == (void*)-1)
		fatal("in shmat 2");
	if ((addr3 = shmat(fd3,NULL,0)) == (void*)-1)
		fatal("in shmat 3");

	unsigned int *bak_info = addr1;
	unsigned int *deliver_info = addr2;
	*bak_info = 0;
	*deliver_info = 0;
	
	pending.START = addr3;
	pending.head = NULL;
	pending.end = NULL;
	pending.offset = 0;
	
	/* ---------------------------------------------------------------------------------
	 * Setting semaphores:
	 * - sem1 for the bakers status shared memory
	 * - sem2 for the deliveries status shared memory
	 * - sem3 for the shared_memory that holds the pending list
	 * - sem4 for the shared_memory that holds the done list for delivery
	 * ---------------------------------------------------------------------------------
	 */

	sem1 = sem_open(SEM_BAKERS,O_CREAT|O_RDWR,S_IRUSR|S_IWUSR, 1);
	if (sem1 == SEM_FAILED)
		fatal("in sem_open 1");
	sem2 = sem_open(SEM_DELIVERIES,O_CREAT|O_RDWR,S_IRUSR|S_IWUSR, 1);
	if (sem2 == SEM_FAILED)
		fatal("in sem_open 2");
	sem3 = sem_open(SEM_ORD_PEND,O_CREAT|O_RDWR,S_IRUSR|S_IWUSR, 1);
	if (sem3 == SEM_FAILED)
		fatal("in sem_open 3");

	
	/* --------------------------------------------------------------------------------
	 * Server starts and accepts connections
	 * --------------------------------------------------------------------------------
	 */
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
			int recv_len = (NPIZZAS+2)*sizeof(char);
			recv(sockfd,&buffer,recv_len,0);
			printf("%s\n",buffer);
		
			/* conversion to int codes */
			int i = 0;
			int codes[NPIZZAS];
			for(i;i<strlen(buffer)-1;i++){
				codes[i] = buffer[i]-'0';
				//printf("%d ",codes[i]);
			}

			/* order parser */
			//printf("length of buffer %d\n",strlen(buffer));
			int type = buffer[strlen(buffer)-1]-'0';
			//printf("%d\n",type);
			int num_pizzas=strlen(buffer)-1;
			order_info *addr = insert(sockfd,num_pizzas,&pending);
			
			int c=num_pizzas;
			//printf("process %d\n",getpid());
			while(c>0){
				if (*bak_info<NBAKERS){
					int bakerpid = fork();
					sem_wait(sem1);
					(*bak_info)++;
					sem_post(sem1);
					if (bakerpid==0){ /* baker process */
						int pizzatype=codes[c-1];
						//printf("pizza time: %d\n",timeofPizza[pizzatype-1]);
						usleep(timeofPizza[pizzatype]*1000);
						sem_wait(sem3);
						addr->status-=1;
						sem_post(sem3);
						printf("I'm %d and i just baked pizza of type %d\n",getpid(),pizzatype);	
						sem_wait(sem1);
						(*bak_info)--;
						sem_post(sem1);
						sem_close(sem1);
						sem_close(sem2);
						sem_close(sem3);
						exit(0);
					}
					/* i reduce the c before the pizza is ready(addr->status)
					 * so that the next baker reads correct pizzatype
					 * doesn't affect the time of pizza because the actual state
					 * of the order is in shared memory (addr->status)
					 */
					c-=1;
				}
			}
			//constantly checks its shared memory for ready status
			int ready=0;
			while(!ready){
				if (addr->status==0)
					ready=1;
			}
			printf("DONE\n");
			sem_close(sem1);
			sem_close(sem2);
			sem_close(sem3);
			exit(0);
		}//Here terminates the cild

		close(sockfd);

	}

	/* detach and delete the 3 shm */
	shmdt(addr1);
	shmctl(fd1,IPC_RMID,NULL);
	shmdt(addr2);
	shmctl(fd2,IPC_RMID,NULL);
	shmdt(addr3);
	shmctl(fd3,IPC_RMID,NULL);
	sem_close(sem1);
	sem_close(sem2);
	sem_close(sem3);
	sem_unlink(SEM_BAKERS);
	sem_unlink(SEM_DELIVERIES);
	sem_unlink(SEM_ORD_PEND);
}

void sig_chld( int signo) {
       pid_t pid;
       int stat;

       while ( ( pid = waitpid( -1, &stat, WNOHANG ) ) > 0 ) {
              printf( "Child %d terminated.\n", pid );
        }
 }

/* ----------------------------------------------------------------------------
 * Functions to manage the queue of free memory space
 *	
 *		front             rear
 *		 |##| <- |##|  <- |##|
 * ---------------------------------------------------------------------------
 */
typedef struct queue{
	int data;
	struct queue *next;
}node;
node *front=NULL;
node *rear=NULL; 

void insert_front(int s){
	node *new;
	new = (node *)malloc(sizeof(node));
	new->data = s;
	new->next = NULL;
	if (front == NULL)
		rear=new;
	else
		front->next=new;
	front=new;	
}

int remove_rear(){
	int ret;
	if (rear!=NULL){
		node *temp=rear;
		rear = temp->next;
		ret=temp->data;
		free(temp);
	}else
		ret=-1;
	return ret;
} 

/* -----------------------------------------------------------------------------
 * Functions for handling the list of orders in shared memory 3
 * - insert
 * - delete
 * - list of free shared memory
 * -----------------------------------------------------------------------------
 */

/* Args:-sockfd of order's client
 * 		-status of baked pizzas of the order_info
 * 		-flag to choose between shared memory/list of pending pizzas and ready pizzas
 * */
order_info *insert(int sockfd,unsigned int status,list_info *list){
	/* dynamic choice of free shared memory 
	 * if a node gets deleted its address is pushed in a queue 
	 * the next time i write in shared memory i choose first a free address
	 * from the queue, if null i choose the next segment pointed by the offset	 *
	 */
	int place;	
 	if((place = remove_rear())==-1)
		place = (list->offset)++;		
	
	order_info *addr = list->START + place*sizeof(order_info);
	addr->client_soc=sockfd;
	addr->status=status;
	if (list->head==NULL){//if list is empty
		list->head=addr;
		list->end=addr;
	}else{
		list->end->next=addr;
		addr->prev=list->end;
		list->end=addr;
	}
	return addr;
}

/* Given a pointer to the order_info object shared memory, memset the memory to 0,
 * fixes the pointers, adds the shared_memory section to the list of free sections
 * */

