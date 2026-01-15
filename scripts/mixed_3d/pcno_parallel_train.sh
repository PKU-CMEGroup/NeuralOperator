#!/bin/bash
#SBATCH -o logs/PCNO_parallel_train_vertex_centered.out
#SBATCH --qos=low
#SBATCH -J PCNO_parallel_train_vertex_centered
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=32
#SBATCH --gres=gpu:2
#SBATCH --time=100:00:00

module load conda
source activate pytorch

export MASTER_ADDR=$(hostname)   # 主节点地址
export MASTER_PORT=29500         # 主节点端口
export NCCL_DEBUG=INFO           # 可选：查看NCCL通信信息

echo "Starting distributed training on $(hostname)"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"


torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
                                    pcno_geo_mixed_3d_parallel_train.py \
                                    --grad True \
                                    --geo True \
                                    --geointegral True \
                                    --num_grad 1 \
                                    --k_max 16 \
                                    --batch_size 5 \
                                    --epochs 500 \
                                    --n_train 1000 \
                                    --n_test 100 \
                                    --to_divide_factor 1.0 \
                                    --mesh_type "vertex_centered" \
                                    > logs/PCNO_parallel_mixed_3d_grad_geo_vertex_centered.log
