#!/bin/sh

usage(){
cat << EOF
Usage: $0 [OPTIONS] {amount of clients}

OPTIONS
-a grep output for how many cokes were given
-b grep output for how many clients got colas
-v grep output for verbose representation of colas distribution on clients 

NOTE
For easy monitoring of the server process
	watch -n 0.1 "ps -ejH|grep ' server'"

EOF
}

clients(){
	for i in `seq 1 $1`
	do
		./client 1 &
	done

	wait $(jobs -p)	
}

#if argument is not supplied, call usage
[[ $# -eq 0 ]] && usage

#if one argument is supplied without options
if [[ $# -eq 1	]]; then
	clients ${@: -1}
else
	output=$(clients ${@: -1});

	while getopts :abv opt; do
		case $opt in
			a) 	echo  -n 'Coca colas given: ';
			   	echo "$output"|grep cola|wc -l ;;

			b) 	echo  -n 'Clients that got colas: ';
				echo "$output"|grep cola|uniq -c|wc -l ;;
		
			v) 	echo "$output"|grep cola|uniq -c ;;

	   		 \?) Invalid option -$OPTARG >&2 ;;
		esac
	done
fi

