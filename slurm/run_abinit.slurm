#!/bin/bash
#SBATCH --job-name=abinit
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=RM
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --time=24:00:00
#SBATCH --output=%x.out

# Run one ABINIT input on Bridges-2 (PSC). Submit from the repository root:
#   sbatch --job-name=er_ko_relax_spin \
#          --export=INPUT=abinit/production/er_ko_relax_spin.abi slurm/run_abinit.slurm
#
# The job runs in abinit/<name>_run/, so the relative paths in every input
# (../common_parameters.abi, ../production/positions_*.abi,
# ../../pseudopotentials/) resolve. The 128-atom calculations used two
# 128-core nodes; the reference cells need far fewer.

set -eu
module load intel-oneapi
module load Abinit/10.4.7-intel
export OMP_NUM_THREADS=1

name=$(basename "$INPUT" .abi)
run="abinit/${name}_run"
mkdir -p "$run"
cp "$INPUT" "$run/${name}.abi"
cd "$run"
mpirun -n "$SLURM_NTASKS" abinit "${name}.abi" > "${name}.log" 2>&1
