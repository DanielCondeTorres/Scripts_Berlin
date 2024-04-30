#!/bin/bash
#SBATCH -n 4 #16
#SBATCH -c 16 #4
#SBATCH --ntasks-per-node=4
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=240G
##SBATCH --mem-per-cpu=240GB
#SBATCH --exclusive

module load cesga/2022 miniconda3/22.11.1-1
##conda activate /mnt/netapp2/Store_uni/home/usc/cq/dct/.conda/envs/envname
conda activate /mnt/netapp2/Store_uni/home/usc/cq/dct/.envi/cg310 #/mnt/lustre/scratch/nlsas/home/usc/cq/dct/.conda2/envs/py36
#module load cesga/system pytorch/1.13.1-cuda
#python ../scripts/gen_input_data.py process_raw_dataset --config trpcage.yaml > paso_1.txt
#python ../scripts/gen_input_data.py build_neighborlists --config trpcage.yaml --config trpcage_priors.yaml > paso_2.txt
#python ../scripts/fit_priors.py compute_statistics --config trpcage_stats.yaml --config trpcage_priors.yaml > paso_3.txt
#python ../scripts/fit_priors.py fit_priors --config trpcage_fit.yaml > paso_4.txt

python ../scripts/gen_sim_input.py --config mhc_2.yaml --config trpcage_priors.yaml > paso_6.txt
