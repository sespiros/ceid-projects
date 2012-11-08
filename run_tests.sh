#!/bin/bash

for i in {1..2}
do
	./client 1 & 
done
wait $(jobs -p)	
