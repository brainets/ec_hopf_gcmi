#!/bin/bash 
#SBATCH --job-name=GC_EC
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=jobid-%A_%a.out
#SBATCH --error=jobid-%A_%a.err
#SBATCH --array=1-100

#Load Matlab 2017a module

ml MATLAB

matlab -nojvm -nodisplay<<-EOF

for condition=1:2
reaction=1;
eff_hopf_gcmi(${SLURM_ARRAY_TASK_ID},reaction,condition);
end


EOF
