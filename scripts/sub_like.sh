#!/bin/bash


for i in `seq 0 9`; do  smart_mpi 1 "python bin/run_like.py production -i $i " -t 12 --include gen6 ; done
