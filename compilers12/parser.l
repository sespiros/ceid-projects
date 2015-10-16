%{		
#include <stdio.h>
#include "parser.tab.h"
int lineNum = 0;
%} 
%option noyywrap

%%
"program"			return RW_PROGRAM;
"var"				return RW_VAR;
"procedure"			return RW_PROCEDURE;
"function"			return RW_FUNCTION;
"if"				return RW_IF;
"then"				return RW_THEN;
"else"				return RW_ELSE;
"while"				return RW_WHILE;
"do"				return RW_DO;
"begin"				return RW_BEGIN;
"end"				return RW_END;
"integer"|"boolean"		return DATATYPE;
":="				return ASSIGN;
[0-9]+				return INT;
"("				return '(';
")"				return ')';
";"				return ';';
":"				return ':';
","				return ',';
"."				return '.';
"="         			return EQ;  
"<>"        			return NE;  
"<"         			return LT;  
">"         			return GT;  
">="        			return GE;  
"<="        			return LE;  
"*"         			return TIMES;  
"div"       			return DIV;  
"mod"       			return MOD;  
"or"        			return OR;  
"and"       			return AND;  
"+"         			return PLUS;  
"-"         			return MINUS; 
"not"				return NOT;
[a-zA-Z][a-zA-Z0-9]*		return ID;
"\n"				{lineNum++;}
"\t"|" "			{}
.				return UNKNOWN_TOKEN;

%%
