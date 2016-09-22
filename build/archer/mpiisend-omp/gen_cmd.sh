#!/bin/sh

for i in  1 2 4 8 # 16 32 64 128 200
do
	Q=qprod;
	OUTPUTFILE=bpmf_${i}_mpiisend.cmd
	cat bpmf.cmd.tmpl | sed -e "s/@NN@/$i/g" > $OUTPUTFILE
done
