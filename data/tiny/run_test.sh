#!/bin/sh

set -e

rm -fr output bpmf_* 
mkdir output

bpmf -r -k -i 9 -b 0 -v -n train.mtx -p test.mtx -o output/ 

RMSE=$(grep "Final Avg RMSE" bpmf_0.out | cut -d : -f 2)

if [ $(echo "$RMSE < 3" |bc -l) -eq 1 ]
then 
	echo OK
	exit 0
else
	echo FAILED
	exit -1
fi

