#include <stdio.h>
#include <lib.h>
#include <unistd.h>

int main()
{
message m;

while(1){
	_syscall(MM, 70, &m);

	printf("Holes: %d\tAverage Size: %d MB\n", m.m3_i1, m.m3_i2);

	sleep(1);
	}
}

