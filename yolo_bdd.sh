#!/bin/bash
#SBATCH --account=PAS2119
#SBATCH --job-name=yolop_8_node
#SBATCH --output=yolop_8_bdd.out
#SBATCH --time=96:00:00

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1 # Cores per node # OSC always give full nodes for parallel jobs
#SBATCH --gpus-per-node=1
#SBATCH --exclusive

# pitzer: 42 dual V100 w/32GB + 32 dual V100 w/16GB + 4 quad V100 w/32GB
# owens: 160 P100 w/16GB
# sinfo -o "%P %G %D %N" | grep gpuparallel to check gpu gres
# ex: gpu:v100-32g, gpu:v100, gpu:v100-quad are on pitzer
##SBATCH --gres=gpu:v100:1 # specify particular gpus
##SBATCH --gres=gpu:p100:1 # specify particular gpus

#SBATCH --mail-type=END
#SBATCH --mail-user=wu.43550@osu.edu

# set -x # each command in the batch file to be printed to the log file as it is executed
source /users/PAS2312/wtywty2001/miniconda/bin/activate
conda activate yolo
export PYTHONNOUSERSITE=true
source /fs/ess/PAS2312/owens/load_mv2
module load cuda/11.6.1
module load mvapich2

###### distributed training ######
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export dist_url="tcp://${master_addr}:${master_port}"
export MASTER_PORT=36666
export WORLD_SIZE=8 # gpus/node * num_nodes
export MASTER_ADDR="${master_addr}"

# export WORLD_SIZE=4 # gpus/node * num_nodes

### get the first node name as master address
echo "NODELIST="${SLURM_NODELIST}
export gpu-per-node=$SLURM_GPUS_PER_NODE
echo "nodes="$SLURM_JOB_NUM_NODES "gpu-per-node="$SLURM_GPUS_PER_NODE "dist_url="$dist_url
###### distributed training ######


cd $SLURM_SUBMIT_DIR
srun python  tools/train.py
echo "Job done. "
