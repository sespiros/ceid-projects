#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/times.h>
#include <stdint.h>

static time_t real_start;
static time_t real_end;

static struct tms start_sys;
static struct tms end_sys;
static clock_t start;
static clock_t end;

void start_clock(void) {
	time(&real_start);
	start = times(&start_sys);
}

void end_clock(void) {
	time(&real_end);
	end = times(&end_sys);
}

void print_clock_results(void) {
	/* real, user, system */
	printf("%i, %d, %d \n",
		(int)(real_end - real_start),
    (intmax_t)(end_sys.tms_utime - start_sys.tms_utime),
		(intmax_t)(end_sys.tms_stime - start_sys.tms_stime));
}

int main(int arg, char **argv) {
    int i = 0;
    int j = 0;
    char *string;
	
    start_clock();

    if (arg>1)
        string = argv[1];
    else {
        printf("Usage: ./cpu_bound word\n");
        return 1;
    }

    while( i++ < 10000000 ) {
       for( j = 0; j < strlen(string)/2; j++ ) {
           if ( string[j] != string[strlen(string)-1-j])
               break;
       }

      /*if ( j == strlen(string)/2)
           printf("it is a palindrome\n");
       else
           printf("it is not a palindrome\n");
       */
    }
	
    end_clock();
    print_clock_results();	
    return 0;
}

