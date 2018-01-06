#!/bin/bash


for i in `seq 0 25`; do  smart_mpi 1 "python bin/run_like.py production_v8 -i $i " -t 12 --include gen4,gen5 ; done
