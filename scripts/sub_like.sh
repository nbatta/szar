#!/bin/bash


#for i in `seq 0 20`; do  smart_mpi 1 "python bin/run_like.py production_v10 -i $i " -t 12 --include gen6 ; done
for i in `seq 0 20`; do  smart_mpi 1 "python bin/run_like.py production_v11_lowbias -i $i " -t 12 --include gen4,gen5 ; done
