#!/bin/bash


for i in `seq 10 25`; do  smart_mpi 1 "python bin/run_like.py production -i $i " -t 12 --include gen4,gen5 ; done
