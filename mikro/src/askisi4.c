#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>

#include "header.h"

#define TC0_ID 17 // for interrupt handler
#define PIOA_ID 2

//input bit
#define BUTTON_START (1<<8)
#define BUTTON_STOP (1<<7)

// led bits
#define F3 (1<<6)
#define F1_RED (1<<5)
#define F1_GREEN (1<<4)
#define F2_RED (1<<3)
#define F2_YELLOW (1<<2)
#define F2_GREEN (1<<1)

#define BUT_RELEASED 0
#define BUT_PRESSED 1

void FIQ_handler (void);

void init_pioa (void) ;
void init_aic (void) ;
void init_tc (void) ;
void clear_aic (void) ;
void clear_tc (void) ;

// new functions

PIO *pioa = NULL ;
AIC *aic = NULL ;
TC *tc = NULL ; 

unsigned int button_start_state = BUT_RELEASED;
unsigned int button_stop_state = BUT_RELEASED;
unsigned int state = 0;
unsigned int request_service = 0;

//main function
int main(void) {

	char key ;
	
	STARTUP;

	init_pioa() ;
	init_tc() ; 
	init_aic() ;

	while ( (key = getchar() != 'e')){

	}

	clear_aic();
	clear_tc() ;
	CLEANUP;

	//and exit
	return 0 ;
}

// function to initialize PIOA
void init_pioa(void) {
	unsigned int tmp;

	//button
	pioa->PER 		= 	BUTTON_START|BUTTON_STOP;
	pioa -> PUER 	= 	BUTTON_START|BUTTON_STOP ; 	// Internal resistor enable for input port
	pioa -> ODR 	= 	BUTTON_START|BUTTON_STOP  ; 	// Input Enable
	pioa -> IER 	= 	BUTTON_START|BUTTON_STOP ;   //enables interrupts for port 9

	//7-segment
	pioa->PER = 0x3F ;
	pioa->OER = 0x3F;
	pioa->CODR = 0x3F;
	
	tmp = pioa -> ISR;			// clear ISR
}

// function to initialize timer
void init_tc(void) {

	unsigned int tmp ;

	tc -> Channel_0.RC = 2048 ; 	// 4Hz
	tc -> Channel_0.CMR = 0x2084 ; 	// Slow clock, count, stop on RC compare
	tc -> Channel_0.IDR = 0xFF ; 	//
	tc -> Channel_0.IER = 0x10 ; 	// TC0 Interrupt Enable

	tmp = tc->Channel_0.SR ; 		// clear any pending timer IRQ

	tc -> Channel_0.CCR = 0x5;		
}

// function to initialize aic
void init_aic(void) { 
	aic -> FFER = AIC(TC0_ID) | AIC(PIOA_ID); // TCO IRQ is FIQ
	aic -> IECR = AIC(TC0_ID) | AIC(PIOA_ID); // Activate TCO IRQ
	aic -> ICCR = AIC(TC0_ID) | AIC(PIOA_ID); // Clear possible pending IRQ
}

void clear_tc(void) {
	tc -> Channel_0.CCR = TIMER_STOP ; // stops the timer
}

void clear_aic(void) {
	aic -> IDCR = AIC(TC0_ID) | AIC(PIOA_ID); // Interrupt disable
}

//FIQ handler
void FIQ_handler(void) {

	unsigned int data_in = 0;
	unsigned int fiq = 0 ;
	unsigned int data_out;
	unsigned int counter;


	fiq = aic -> IPR ; 					// read the IRQ source

	if (fiq & (1<<PIOA_ID)) {
		data_in = pioa->ISR;			//clear the interrupt source of pioa
		aic->ICCR = AIC(PIOA_ID);		//then clears the interrupt from aic

		data_in = pioa->PDSR;			//read input of PDSR (line 1)
        
        if ( data_in & BUTTON_START ) { 	
	     	if ( button_start_state == BUT_RELEASED) {
	     		button_start_state = BUT_PRESSED;
	     		request_service = 1;
			}
		} else {
			if (button_start_state == BUT_PRESSED) {
				button_start_state = BUT_RELEASED;
			}
		}

		if ( data_in & BUTTON_START ) { 	
	     	if ( button_stop_state == BUT_RELEASED) {
	     		button_stop_state = BUT_PRESSED;
	     		request_service = 0;
			}
		} else {
			if (button_stop_state == BUT_PRESSED) {
				button_stop_state = BUT_RELEASED;
			}
		}
	}

	if ( fiq & (1<<TC0_ID) ) {		//if source of interrupt is 17 timer/counter 0

		data_out = tc -> Channel_0.SR ; 	// clear timer IRQ
		aic -> ICCR = (1<<TC0_ID) ; 		// then clear pending from aic
         
        switch (state){
        case 0:
        	pioa->CODR = F2_RED;
            pioa->SODR = F1_RED|F2_GREEN;

        	state = ( request_service == 1 )?1:0;
        	counter = 0;
        	break;

        case 1:
        	counter++;
        	if( counter == 4*10 ){
        		state = 2;
        		counter = 0;
        	}
        	break;

        case 2:
        	pioa->CODR = F2_GREEN;
        	pioa->SODR = F2_YELLOW;

        	counter++;
        	if ( counter == 4*3 ){
        		state = 3;
        		counter = 0;
        	}
        	break;

        case 3:
            pioa->CODR = F2_YELLOW;
        	pioa->SODR = F2_RED;

        	counter++;
        	if ( counter == 4*2 ){
        		state = 4;
        		counter = 0;
        	}
        	break;

        case 4:
            pioa->CODR = F1_RED;
        	pioa->SODR = F1_GREEN;

        	counter++;
        	if ( counter == 4*10 ){
        		state = 5;
        		counter = 0;
        	}
        	break;

        case 5:
        	state = ( request_service == 0 )?6:0;
        	break;

        default:
            pioa->CODR = F1_GREEN;
        	pioa->SODR = F1_RED;

        	counter++;
        	if ( counter == 4*5 ){
        		state = 0;
        		counter = 0;
        	}
		}

		if( counter/2 == 0 && state != 0 && state != 6 ){
			data_in = pioa->PDSR;
			pioa->SODR = data_in & F3;
			pioa->CODR = ~data_in & F3;
		}else
			pioa->CODR = F3;
			

		tc -> Channel_0.CCR = 0x5 ; 		// restart the timer
	}	
		
}

