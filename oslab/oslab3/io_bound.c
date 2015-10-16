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
	/* real, user, system*/
	printf("%i, %d, %d \n",
        (int)(real_end - real_start),
        (intmax_t)(end_sys.tms_utime - start_sys.tms_utime),
        (intmax_t)(end_sys.tms_stime - start_sys.tms_stime));
}

int main(int arg, char **argv) {
    int i = 0;
    char buffer[100];
    FILE *fp;
	
    start_clock();

    if (arg<=1) {
        printf("Usage: ./io_bound filename\n");
        return 1;
    }

    while( i++ < 100000 ) {
        if (( fp = fopen(argv[1], "r") ) == NULL) {
            printf("File does not exist or couldn't be opened for reading \n");
            return 1;
        }
        fread(buffer, 100, 1, fp);
	fseek(fp, 0, SEEK_SET);
        fclose(fp);
    }

    end_clock();
    print_clock_results();
    return 0;
}

