%{		
#include <stdio.h>
#include "parser.tab.h"

%} 
%option noyywrap

%%
"integer"|"boolean"		 return DATATYPE;
"("						 return PAR_L;
"("						 return PAR_R;
[0-9]+					 return INT;
"="         			 return EQ;  
"<>"        			 return NE;  
"<"         			 return LT;  
">"         			 return GT;  
">="        			 return GE;  
"<="        			 return LE;  
"*"         			 return TIMES;  
"div"       			 return DIV;  
"mod"       			 return MOD;  
"or"        			 return OR;  
"and"       			 return AND;  
"+"         			 return PLUS;  
"-"         			 return MINUS; 
"not"				     return NOT;
[a-zA-Z][a-zA-Z0-9]* 	 return ID;
.;

%%
