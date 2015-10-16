#include <sys/types.h>
#include <sys/stat.h>
#include <fcnt1.h>
#include <sys/ioct1.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include "header.h"


/*
*This example makes PIOA bit 14 an output
*and periodically writes 0 and 1 to this bit
*This makes a LED switching on and off
*/

#define PIO(x) ((1)<<(x))

//output bit to drive 

#define OBIT 14
#define TC0_ID 17

void FIQ_handler (void);

void init_pioa (void) ;
void init_aic (void) ;
void init_tc (void) ;

void clear_aic (void) ;
void clear_tc (void) ;

void led_on();
void led_off() ;
unsigned int led_is_off(void) ;

unsigned int nfiq=0 ;

//Declare pointers to peripherals
//All peripherals are needed for the STARTUP macro
PIO *pioa = NULL ;
AIC *aic = NULL ;
TC *tc = NULL ; 

//main function
int main(void) {

	char key ; 

	//1. system initialazation
	STARTUP;

	//2. initialise peripherals
	init_pioa() ;

	//3. initially turn led off
	led_off() ;

	//4. initialize timer
	init_tc() ; 

	//5. initialize interrupt controller
	init_aic() ;

	tc -> Channel_0.CCR = 0x5; //6. start the timer

	//7. wait until the user presses 'x'
	while ( (key = getchar() != 'x'))
	{}

	//8. then cleanup peripherals and system
	clear_aic();
	clear_tc() ;
	led_off() ; //9. turn off led
	CLEANUP;

	//and exit
	return 0 ;
}

// function to initialize PIOA
void init_pioa(void) {
	pioa -> PER = PIO(OBIT) ; // General Purpose Enable
	pioa -> OER = PIO(OBIT) ; // Output Enable

}

// function to initialize timer
void init_tc(void) {

	unsigned int tmp ;

	tc -> Channel_0.RC = 4096 ; //
	tc -> Channel_0.CMR = 0x2084 ; // 
	tc -> Channel_0.IDR = 0xFF ; //
	tc -> Channel_0.IER = 0x10 ; // TC0 Interrupt Enable

	tmp = tc->Channel_0.SR ; // clear any pending timer IRQ

}

// function to initialize aic
void init_aic(void) { 
	aic -> FFER = (1<<TC0_ID) ; // TCO IRQ is FIQ
	aic -> IECR = (1<<TC0_ID) ; // Activate TCO IRQ
	aic -> ICCR = (1<<TC0_ID) ; // Clear possible pending IRQ

}

void clear_tc(void) {
	tc -> Channel_0.CCR = 0x02 ; 
}

void clear_aic(void) {
	aic -> IDCR = (1<<TC0_ID) ; // Interrupt disable
}

//FIQ handler
void FIQ_handler(void) {

	unsigned int tmp ; 
	unsigned int fiq = 0 ;


	fiq = aic -> IPR ; // read the IRQ source

	if (fiq & (1<<TC0_ID)) {

		tmp = tc -> Channel_0.SR ; // clear timer IRQ
		tc -> Channel_0.CCR = 0x5 ; // start the timer
		
		aic -> ICCR = (1<<TC0_ID) ; // Clear pending Timer IRQ

		if (nfiq&1)
			led_on() ;
		else
			led_off() ;

		nfiq++ ;
	}
		
}

void led_on() {
	pioa->CODR = PIO(OBIT); // clear the bit
}

void led_off() {
	pioa->SODR = PIO(OBIT); // set the bit
}