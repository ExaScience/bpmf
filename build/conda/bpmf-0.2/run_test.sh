#!/bin/sh

for bin in $PREFIX/bin/bpmf-*
do
	! $bin 
done
