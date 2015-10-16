/*
***********************************
*         Δεύτερο ερώτημα
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
#define AIC(x) ((1)<<(x))

//output bit to drive 
#define IBIT 7
#define TC0_ID 17 // for interrupt handler
#define PIOA_ID 2

#define BUT_IDLE 0
#define BUT_PRESSED 1

#define LED_IDLE 0
#define LED_LEFT 1
#define LED_RIGHT 2

void FIQ_handler (void);

void init_pioa (void) ;
void init_aic (void) ;
void init_tc (void) ;

void clear_aic (void) ;
void clear_tc (void) ;

unsigned int button_state = BUT_IDLE;
unsigned int led_state = LED_IDLE;
int mask;

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
	init_tc() ; 
	init_aic() ;

	counter = 100;
	mask = 1;
	while ( (key = getchar() != 'e'))
	{
		counter+=1;
		if(counter>=100){
			pioa -> CODR = 0x7F; // turn off leds
			bitmask = mask;
			next = 20;
			counter = 0;
		}else{
			if(counter>=next && counter<=80) {
				pioa -> SODR = bitmask;
				bitmask<<=1;
				if(bitmask==0x80)
					bitmask = 1;
				next+=10;
			}
		}
	}

	clear_aic();
	clear_tc() ;
	CLEANUP;

	//and exit
	return 0 ;
}

// function to initialize PIOA
void init_pioa(void) {
	unsigned int gen;
	unsigned int i;
	pioa -> PER = 0;
	pioa -> OER = 0;
	pioa -> CODR = 0;
	for(i=0; i<7; i++) 
	{
		pioa -> PER |= PIO(i) ; // General Purpose Enable for outputs

		pioa -> CODR |= PIO(i) ; 	// set output to 0 
		pioa -> OER |= PIO(i) ; 	// Output Enable
	}

	pioa -> PER |= PIO(IBIT); // General Purpose Enable for input
	pioa -> PUER = PIO(IBIT) ; 	// Internal resistor enable for input port
	pioa -> ODR = PIO(IBIT)  ; 	// Input Enable

	
	gen = pioa -> ISR;			// clear ISR

	pioa -> IER = PIO(IBIT) ;   //enables interrupts for port 7
}

// function to initialize timer
void init_tc(void) {

	unsigned int tmp ;

	tc -> Channel_0.RC = 1638 ; 	// 5Hz
	tc -> Channel_0.CMR = 0x2084 ; 	// Slow clock, count, stop on RC compare
	tc -> Channel_0.IDR = 0xFF ; 	//
	tc -> Channel_0.IER = 0x10 ; 	// TC0 Interrupt Enable

	tmp = tc->Channel_0.SR ; 		// clear any pending timer IRQ

}

// function to initialize aic
void init_aic(void) { 
	aic -> FFER = AIC(TC0_ID) | AIC(PIOA_ID); // TCO IRQ is FIQ
	aic -> IECR = AIC(TC0_ID) | AIC(PIOA_ID); // Activate TCO IRQ
	aic -> ICCR = AIC(TC0_ID) | AIC(PIOA_ID); // Clear possible pending IRQ

}

void clear_tc(void) {
	tc -> Channel_0.CCR = 0x02 ; // stops the timer
}

void clear_aic(void) {
	aic -> IDCR = AIC(TC0_ID) | AIC(PIOA_ID); // Interrupt disable
}

//FIQ handler
void FIQ_handler(void) {

	unsigned int data_in = 0;
	unsigned int fiq = 0 ;
	unsigned int data_out;


	fiq = aic -> IPR ; 				// read the IRQ source

	if (fiq & (1<<PIOA_ID)) {
		data_in = pioa->ISR;		//clear the interrupt source of pioa
		aic->ICCR = AIC(PIOA_ID);	//then clears the interrupt from aic

		data_in = pioa->PDSR;		//read input of PDSR (line 1)
		if ( data_in & IBIT ) {		//button pressed
			if ( button_state == BUT_IDLE ) {
				button_state = BUT_PRESSED;
				if ( led_state == LED_IDLE ) {
					led_state = LED_RIGHT ;
					tc -> Channel_0.CCR = 0x5 ; // start the timer
				}else if( led_state == LED_RIGHT ) {
					led_state = LED_LEFT ;
					tc -> Channel_0.CCR = 0x5 ; // start the timer
				}else {
					led_state = LED_IDLE;
					clear_tc();
				}
			}
		}else{
			if(button_state == BUT_PRESSED)
				button_state = BUT_IDLE;
		}
	}

	if ( fiq & (1<<TC0_ID) ) {		//if source of interrupt is 17 timer/counter 0

		data_out = tc -> Channel_0.SR ; 	// clear timer IRQ
		aic -> ICCR = (1<<TC0_ID) ; 		// then clear pending from aic

		if( led_state == LED_RIGHT ) {
			mask >>= 1;
			if( mask == 0x00 )
				mask = 0x40;
		}else if( led_state == LED_LEFT ) {
			mask <<= 1;
			if( mask == 0x80 )
				mask = 1;
		}

		tc -> Channel_0.CCR = 0x5 ; 		// start the timer
	}	
		
}
