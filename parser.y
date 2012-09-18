%{
#include <stdio.h>
extern FILE *yyin;
extern FILE *yyout;
void yyerror (char *);
%}

%token  ID
%token  INT
%token  DATATYPE
%token	PAR_L PAR_R
%left   OR
%left   AND
%left   NOT
%left   LT LE GT GE NE EQ
%left   PLUS MINUS
%left   TIMES MOD DIV 

%%
program:
	   program ID ; block .
	   ;

block: PAR_L local-definition PAR_R

expression:
	expression binary_op expression
	| unary_op expression
	| PAR_L expression PAR_R
	| INT
	| ID
	;

binary_op :
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

unary_op :
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
