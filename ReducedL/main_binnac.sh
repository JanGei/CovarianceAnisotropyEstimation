#!/bin/bash

### Job Name
#PBS -N EnKF_n_280

### Set email type for job

### Accepted options: a (abort), b (begin), e (end)
#PBS -m abe

### email address for user

#PBS -M janek.geiger@uni-tuebingen.de

### Queue name that job is submitted to
#PBS -q long  

### Request resources
#PBS -l nodes=1:ppn=28
#PBS -l walltime=7:00:00:00

echo Running on host `hostname`
echo Time is `date`

# Load conda environment
# This is needed to find conda on the nodes
source /home/tu/tu_tu/tu_epajg01/miniconda3/etc/profile.d/conda.sh
conda activate Corrl
which python

module load geo/modflow/6.4.2
# run Python file
python /beegfs/work/tu_epajg01/Python/Corellation/EnKF_280/main.py

# print some diagnostic output
echo $PBS_O_WORKDIR/
echo $PBS_O_HOST
echo $PBS_QUEUE
echo $PBS_NODEFILE
echo $TMPDIR
