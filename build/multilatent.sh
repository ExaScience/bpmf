#!/bin/sh

for NL in 8 10 16 32 64 100 128
do
	make clean
	make -j BPMF_NUMLATENT=$NL
	mv bpmf bpmf-$NL
done
