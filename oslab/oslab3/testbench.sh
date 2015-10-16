#####################################
# 5070 Seimenis Spyros
# 4993 Kallivokas Dimitris
#####################################

#!/usr/bin/bash

rm -rf output
mkdir output
touch output/cpu_times.txt
touch output/io_times.txt

./cpu_bound racecar &
./io_bound cpu_bound.c &
wait

rm cpu_times.txt
rm io_times.txt
for((j=0 ; j < $1 ; j++))
do
	echo Loop$j --------- >> output/cpu_times.txt
	echo Loop$j --------- >> output/io_times.txt
	for i in 1 2 3 
	do
		./cpu_bound racecar    1>> output/cpu_times.txt&
		./io_bound cpu_bound.c 1>> output/io_times.txt&
	done
	wait
done	

echo "Testbench done!"


