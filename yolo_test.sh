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
export WORLD_SIZE=2 # gpus/node * num_nodes
export MASTER_ADDR="tcp://${master_addr}:"

### get the first node name as master address
echo "NODELIST="${SLURM_STEP_NODELIST}
echo "nodes="$SLURM_JOB_NUM_NODES "gpus-per-node="$SLURM_GPUS_PER_NODE "dist_url="$dist_url
###### distributed training ######

srun -n 2   --export=ALL,MV2_USE_CUDA=1,MV2_CUDA_BLOCK_SIZE=8388608 python  tools/train.py
# LD_PRELOAD=/fs/ess/PAS2312/owens/mvapich2/lib/libmpi.so srun -n 2   --export=ALL,MV2_USE_CUDA=1,MV2_CUDA_BLOCK_SIZE=8388608 python tools/train.py 
echo "Job done. "
