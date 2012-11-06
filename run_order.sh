#!/bin/bash

for i in {1..100}
do
	./client 1 >/dev/null &
done
