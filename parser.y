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
%token	RW_ELSE
%token  RW_WHILE 
%token  RW_DO 
%token  RW_BEGIN 
%token  RW_END
%token  ID
%token  DATATYPE
%token	ASSIGN
%token	INT
%left   OR
%left   AND
%left   NOT
%left   LT LE GT GE NE EQ
%left   PLUS MINUS
%left   TIMES MOD DIV 
%start program
%%

program: 
	RW_PROGRAM ID ';' block   		{printf("PROGRAM! \n");}
	;

block: 
	many_locals compound_statement		{printf("Block \n");}			
	;

many_locals:
						{printf("Many Locals: NOTHING \n");}
	|local_definition many_locals 		{printf("Many Locals: LocalDef \n");}
	;

local_definition:
	variable_definition        	  	{printf("LocDef : VarDef \n");}
	| procedure_definition			{printf("LocDef : ProcDef \n");}
	| function_definition			{printf("LocDef : FunDef \n");}
	;

variable_definition: 
	RW_VAR def_some_variables ';' more_vars 	{printf("Variable Definition \n");}
	;

more_vars: 

	| def_some_variables ';' more_vars 
	; 

def_some_variables: 
	ID more_IDs ':' DATATYPE		{printf("Defined some vars \n");}
	;

more_IDs:
						{printf("No More IDs \n");}
	| ',' ID more_IDs			{printf("One More ID \n");}
	;

procedure_definition:
	procedure_header block ';'		{printf("Procedure Definition \n");}
	;

procedure_header:
	RW_PROCEDURE ID formal_parameters ';'	{printf("Procedure Header \n");}
	;

function_definition:
	function_header block ';'		{printf("Function Definition \n");}
	;

function_header:
	RW_FUNCTION ID formal_parameters ':' DATATYPE ';'	{printf("Function Header \n");}
	; 

formal_parameters:
								{printf("Formal Params: NOTHING \n");}
	|'(' def_some_variables more_parameters ')' 		{printf("Formal Params: OK\n");}
	;

more_parameters:
								{printf("-a- \n");}
	| ';' def_some_variables more_parameters		{printf("-aaa- \n");}
	;

statement:
	
	| ID assignment_or_proc_func_call		{printf("Statement: Assign OR Proc_Func_Call \n");}
	| if_statement					{printf("Statement: IF \n");}
	| while_statement				{printf("Statement: WHILE \n");}
	| compound_statement				{printf("Statement: Compound \n");}
	;

assignment_or_proc_func_call:
	ASSIGN expression				{printf("Assign \n");}
	| actual_parameters_or_not			{printf("Parameters \n");}
	;		

actual_parameters_or_not:
							{printf("Actual Params: NOTHING \n");}
	|'(' actual_parameters ')'			{printf("Actual Params: (some params)\n");}
	;

if_statement:
	RW_IF expression RW_THEN statement else_statement	{printf("IF statement \n");}
	;

else_statement:

	| RW_ELSE statement					{printf("ELSE statement \n");}
	;

while_statement:
	RW_WHILE expression RW_DO statement		{printf("WHILE statement \n");}
	;

actual_parameters:
	expression ',' actual_parameters		{printf("Actual Params: More Exs \n");}
	| expression					{printf("Actual Params: One Ex \n");}
	;

compound_statement:
	RW_BEGIN statement more_statements RW_END	{printf("Compound Statement \n");}
	;

more_statements:
							{printf("No more statements \n");}
	| ';'statement more_statements			{printf("; One More Statement \n");}
	;	

expression:
	expression binary_op expression			{printf("Ex BinOp Ex \n");}
	| unary_op expression				{printf("UnOp Ex \n");}
	| '(' expression ')'				{printf("( Ex ) \n");}
	| ID actual_parameters_or_not			{printf("ID or Proc_Func_Call \n");}
	| INT						{printf("INT \n");}
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
	yyout = fopen("output","w");
	yyparse();
	return 0;
}
