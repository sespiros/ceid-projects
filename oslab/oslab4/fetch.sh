


if [ $# -gt 0 ]
then
	for i in $@
	do
		scp sespiros@172.16.169.1:~/Dropbox/projects/ceid/oslab4/$i .
	done
else
	echo "Usage: ./fetch.sh file1 file2 ..."
fi

