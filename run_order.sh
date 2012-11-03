#!/bin/bash

for i in {1..10}
do
	./client 1 >/dev/null &
done
