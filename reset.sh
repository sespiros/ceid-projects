#!/bin/sh 

echo "The following shared memory segment will be deleted"
for i in {5070..5073}
do
	s=`ipcs |grep $(echo "obase=16; $i"|bc|tr '[A-Z]' '[a-z]')|awk '{print $2}'`
	if [ -n "$s" ]; then
		echo $s
		ipcrm shm $s
	fi
done
