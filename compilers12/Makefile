CC			= gcc
LEX 		= flex
BISON		= bison
BFLAGS		= -d
EXECUTABLE 	= parser

default:
	$(BISON) $(BFLAGS) $(EXECUTABLE).y 
	$(LEX) $(EXECUTABLE).l
	$(CC) lex.yy.c $(EXECUTABLE).tab.c -o $(EXECUTABLE)

clean:
	rm lex.yy.c
	rm $(EXECUTABLE).tab.h
	rm $(EXECUTABLE).tab.c
