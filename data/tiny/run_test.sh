#!/bin/bash

set -e

rm -fr output bpmf_* 
mkdir output

BPMF=bpmf
[ -n "$1" ] && BPMF=$1

set -x
$BPMF -r -k -i 9 -b 0 -v -n train.mtx -p test.mtx -o output/ 

RMSE=$(grep "Final Avg RMSE" bpmf_0.out | cut -d : -f 2)
CHECK="$RMSE < 3"
[ -n "$2" ] && CHECK="($RMSE - $2) < 0.1 && ($2 - $RMSE < 0.1)"

if [ $(echo "$CHECK" |bc -l) -eq 1 ]
then 
	echo OK
	exit 0
else
	echo FAILED
	exit -1
fi

