


if [ $# -gt 0 ]
then
	for i in $@
	do
		scp sespiros@`redacted IP`:~/Dropbox/projects/ceid/oslab3/$i .
	done
else
	echo "Usage: ./fetch.sh file1 file2 ..."
fi
