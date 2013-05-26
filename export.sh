DIRNAME=streamcluster_omp

rm -rf $DIRNAME
mkdir $DIRNAME
cd $DIRNAME
cp ../doc/tex/report.pdf ../streamcluster_omp.cpp ../Makefile ../README .
cd ..
tar czvf project.tar.gz $DIRNAME/
