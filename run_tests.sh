#!/bin/bash

for i in `seq 1 $1`
do
	./client 1 & 
done
wait $(jobs -p)	
