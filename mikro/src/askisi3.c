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
#define DOT 8

//input bit
#define IBIT 9

#define TC0_ID 17 // for interrupt handler
#define PIOA_ID 2

#define BUT_IDLE 0
#define BUT_PRESSED 1
#define BUT_RELEASED 2

#define LED_IDLE 0
#define LED_ON 1
#define LED_HOLD 2

void FIQ_handler (void);

void init_pioa (void) ;
void init_aic (void) ;
void init_tc (void) ;

void clear_aic (void) ;
void clear_tc (void) ;

unsigned int button_state = BUT_IDLE;
unsigned int led_state = LED_IDLE;

unsigned int counter;

//Declare pointers to peripherals
//All peripherals are needed for the STARTUP macro
PIO *pioa = NULL ;
AIC *aic = NULL ;
TC *tc = NULL ; 

//main function
int main(void) {

	char key ;
	
	STARTUP;

	init_pioa() ;
	init_tc() ; 
	init_aic() ;

	counter = 0;
	while ( (key = getchar() != 'e'))
	{
       // pioa->SODR = counter%10<<4|counter/10;
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
	for(i=0; i<8; i++) 
	{
		pioa -> PER |= PIO(i) ; // General Purpose Enable for outputs

		pioa -> CODR |= PIO(i) ; 	// set output to 0 
		pioa -> OER |= PIO(i) ; 	// Output Enable
	}

	pioa -> PER |= PIO(IBIT); // General Purpose Enable for input
	pioa -> PUER = PIO(IBIT) ; 	// Internal resistor enable for input port
	pioa -> ODR = PIO(IBIT)  ; 	// Input Enable

	
	gen = pioa -> ISR;			// clear ISR

	pioa -> IER = PIO(IBIT) ;   //enables interrupts for port 9
}

// function to initialize timer
void init_tc(void) {

	unsigned int tmp ;

	tc -> Channel_0.RC = 8192 ; 	// 5Hz
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
    unsigned int reset = 0;


	fiq = aic -> IPR ; 					// read the IRQ source

	if (fiq & (1<<PIOA_ID)) {
		data_in = pioa->ISR;			//clear the interrupt source of pioa
		aic->ICCR = AIC(PIOA_ID);		//then clears the interrupt from aic

		data_in = pioa->PDSR;			//read input of PDSR (line 1)
        
        if ( data_in & PIO(IBIT) ) { 	
	     	if ( button_state == BUT_RELEASED) {
	     		button_state = BUT_PRESSED;

				if ( led_state == LED_IDLE ) {
					led_state = LED_ON ;
					tc->Channel_0.CCR = 0x2;		//stop
					data_in = tc->Channel_0.SR;	 	// flush pending interrupts				
		            tc -> Channel_0.RC = 8190 ; 	// 1 second
					tc -> Channel_0.CCR = 0x5 ; 	// start the timer
				}else if( led_state == LED_ON ) {
					led_state = LED_HOLD ;
					tc->Channel_0.CCR = 0x2;		//stop
					data_in = tc->Channel_0.SR; 	// flush pending interrupts	
		            tc -> Channel_0.RC = 8190/2 ; 	// 0,5 second
					tc -> Channel_0.CCR = 0x5 ; 	// start the timer
				}else {
					led_state = LED_IDLE;
					clear_tc();
	                pioa->CODR = PIO(DOT) & PIO(IBIT);
				}
			}
		} else {
			if (button_state == BUT_PRESSED) {
				button_state = BUT_RELEASED;
				reset = 0;
			}
		}
	}

	if ( fiq & (1<<TC0_ID) ) {		//if source of interrupt is 17 timer/counter 0

		data_out = tc -> Channel_0.SR ; 	// clear timer IRQ
		aic -> ICCR = (1<<TC0_ID) ; 		// then clear pending from aic
         
		if( led_state == LED_ON ) {
            counter++;
            if ( counter > 59 )
                counter = 0;
           	pioa->SODR = counter%10<<4|counter/10;

           	if ( button_state == BUT_PRESSED )
           		reset += 2;

		}else if( led_state == LED_HOLD ) {
            if ( pioa->ODSR & PIO(DOT) )
                pioa->CODR |= PIO(DOT);
            else
                pioa->SODR |= PIO(DOT);

            if ( button_state == BUT_PRESSED )
           		reset += 1;

		}

		if (reset > 1) {
			counter = 0;
         	pioa->SODR = counter%10<<4|counter/10;

            reset = 0;
            button_state = BUT_IDLE;
            button_state = BUT_RELEASED;
            //pioa->CODR = 0xff;
		}

		tc -> Channel_0.CCR = 0x5 ; 		// start the timer
	}	
		
}

