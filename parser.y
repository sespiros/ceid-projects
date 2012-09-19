%{
#include <stdio.h>
extern FILE *yyin;
extern FILE *yyout;
void yyerror (char *);
%}

%token  RW_PROGRAM 
%token  RW_VAR 
%token  RW_PROCEDURE 
%token  RW_FUNCTION
%token  RW_IF 
%token  RW_THEN 
%token  RW_WHILE 
%token  RW_DO 
%token  RW_BEGIN 
%token  RW_END
%token  ID
%token  INT
%token  DATATYPE
%left   OR
%left   AND
%left   NOT
%left   LT LE GT GE NE EQ
%left   PLUS MINUS
%left   TIMES MOD DIV 
%start program
%%

program:  expression | variable_definition | function_header
	| procedure_header;

variable_definition: 
	RW_VAR many_vars 
	;

many_vars: 
	def_some_variables ';' many_vars 
	| def_some_variables ';'
	; 

def_some_variables: 
	ID ',' def_some_variables
	| ID ':' DATATYPE 
	;

procedure_header:
	RW_PROCEDURE ID formal_parameters ';'
	;

function_header:
	RW_FUNCTION ID formal_parameters ':' DATATYPE ';'
	;

formal_parameters:
	
	|'(' many_parameters ')' 
	;

many_parameters:
	def_some_variables ';' many_parameters
	| def_some_variables
	;

proc_func_call:
	ID '[''(' actual_parameters ')'']'
	;

actual_parameters:
	expression ',' actual_parameters
	| expression
	;

expression:
	expression binary_op expression	{printf("Ex-BinOp-Ex \n");}
	| unary_op expression		{printf("UnOp-Ex \n");}
	| '(' expression ')'	  	{printf("(-Ex-) \n");}
	| proc_func_call
	| INT
	| ID
	;

binary_op:
	PLUS 
	| MINUS
	| TIMES
	| DIV
	| AND
	| OR
	| LE
	| LT
	| GE
	| GT
	| EQ
	| NE
	| MOD
	;

unary_op:
	NOT
	| PLUS
	| MINUS 
	;

%%

void yyerror(char *s){
	fprintf(stderr,"%s\n",s);
}

int main (int argc, char **argv){
	++argv; --argc;
	if (argc > 0 )
		yyin = fopen(argv[0], "r");
	else
		yyin = stdin;
	yyout = fopen ("output","w");
	yyparse();
	return 0;
}
