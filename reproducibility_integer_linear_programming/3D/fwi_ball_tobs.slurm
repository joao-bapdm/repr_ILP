#!/bin/bash
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
echo -e "\n## Job started at $(date +'%d-%m-%Y as %T') #####################\n"
echo -e "\n## Jobs activated by $USER: \n"
squeue -a --user=$USER
echo -e "\n## Execution node:         $(hostname -s) \n"
echo -e "\n## Number of tasks per job: $SLURM_NTASKS \n"

#########################################
##------- Start of job     ----- #
#########################################
## Configure the execution environment
module purge
module load singularity-3.5.3-gcc-8.3.0-3hixhf5

## Define environment variables
export FIREDRAKE_CACHE_DIR=$PWD/FIREDRAKE_CACHE_DIR_$SLURM_JOBID
export PYOP2_CACHE_DIR=$PWD/PYOP2_CACHE_DIR_$SLURM_JOBID
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=$PWD/FIREDRAKE_TSFC_KERNEL_CACHE_DIR_$SLURM_JOBID
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

## Information about the entry and exit of the job
echo -e "\n## Job submission directory:   $SLURM_SUBMIT_DIR \n"
srun --mpi=pmi2 singularity exec spyro_cplex_20_10.sif python run_forward_cubic_mesh.py \
     -c joao_paper/centered_ball_tobs.json \
     -i joao_paper/centered_ball.hdf5
srun --mpi=pmi2 singularity exec spyro_cplex_20_10.sif python run_fwi_tobs.py \
     -c joao_paper/centered_ball_tobs.json \
     # -i joao_paper/centered_ball.hdf5
echo -e "\n## Job finished on $(date +'%d-%m-%Y as %T') ###################"

## Cleanup cache
rm -rf $FIREDRAKE_CACHE_DIR
rm -rf $PYOP2_CACHE_DIR
rm -rf $FIREDRAKE_TSFC_KERNEL_CACHE_DIR
