/*
* Εργαστήριο μικροεπεξεργαστών
* Άσκηση 2η
* 
* Σειμένης Σπύρος 		5070
* Βιττωράκης Ιωάννης 	4963
* Αβορίτης Κωνσταντίνος	5164
*
*/

/*
***********************************
*         Πρώτο ερώτημα
***********************************
*/
#include <sys/types.h>
#include <sys/stat.h>
#include <fcnt1.h>
#include <sys/ioct1.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>

#include "header.h"

// helper function for acquiring bitfield for a port
#define PIO(x) ((1)<<(x))

void init_pioa (void);

//Declare pointers to peripherals
//All peripherals are needed for the STARTUP macro
PIO *pioa = NULL ;
AIC *aic = NULL ;
TC *tc = NULL ; 

//main function
int main(void) {

	char key ;
	unsigned int counter, next;
	int bitmask;

	STARTUP;

	init_pioa() ;

	counter = 100;
	while ( (key = getchar() != 'e'))
	{
		counter+=1;
		if(counter>=100)
		{
			pioa -> CODR = 0x7F; // turn off leds
			bitmask = 1;
			next = 20;
			counter = 0;
		}else{
			if(counter>=next && counter<=80)
			{
				pioa -> SODR = bitmask;
				bitmask<<=1;
				next+=10;
			}
		}
	}

	CLEANUP;

	//and exit
	return 0 ;
}

// function to initialize PIOA
void init_pioa(void) {
	unsigned int i;
	pioa -> PER = 0;
	pioa -> OER = 0;
	pioa -> CODR = 0;
	for( i=0; i<7; i++ ) 
	{
		pioa -> PER |= PIO(i) ; // General Purpose Enable

		pioa -> CODR |= PIO(i) ; 	// set output to 0 
		pioa -> OER |= PIO(i) ; 	// Output Enable
	}
}
