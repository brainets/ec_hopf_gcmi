#!/bin/bash 
#SBATCH --job-name=GC_EC
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=2
#SBATCH --output=jobid-%A_%a.out
#SBATCH --error=jobid-%A_%a.err
#SBATCH --array=1-3

#Load Matlab 2017a module

ml MATLAB

matlab -nojvm -nodisplay<<-EOF

for condition=1:2
eff_hopf_gcmi(${SLURM_ARRAY_TASK_ID},condition);
end


EOF