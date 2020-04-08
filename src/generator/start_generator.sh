#!/bin/bash
#PBS -l nodes=1:ppn=64
#PBS -l walltime=0:30:00
#PBS -q short-laser

TARGET = "/bigdata/hplsim/production/aipr"
PROCESSES = 64
NUMIMAGES = 100000

cd deeplearning-phaseretrieval/conv_net
source load_env.sh
cd fowgas/packages/synthetic_saxs

mpirun -n PROCESSES python mpi_image_generator.py NUMIMAGES TARGET
