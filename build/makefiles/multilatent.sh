#!/bin/sh

NL="8 16 32 64 128 10 20 30 40 50 60 70 80 90 100"

for K in $NL
do
	make clean
	make -j BPMF_NUMLATENT=$K
	mv bpmf bpmf-$K
done
