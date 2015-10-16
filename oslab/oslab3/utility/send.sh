


if [ $# -gt 0 ]
then
	for i in $@
	do
		scp $i sespiros@`redacted IP`:~/Dropbox/projects/ceid/oslab3
	done
else
	echo "Usage: ./send.sh file1 file2 ..."
fi
