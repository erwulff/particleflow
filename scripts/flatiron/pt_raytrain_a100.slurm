#!/bin/sh

# Walltime limit
#SBATCH -t 168:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --constraint=a100-80gb&sxm4

# Job name
#SBATCH -J pt_raytrain

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"


module --force purge; module load modules/2.2-20230808
module load slurm gcc cmake cuda/12.1.1 cudnn/8.9.2.26-12.x nccl openmpi apptainer

nvidia-smi
source ~/miniconda3/bin/activate pytorch
which python3
python3 --version

export CUDA_VISIBLE_DEVICES=0,1,2,3
num_gpus=${SLURM_GPUS_PER_TASK}  # gpus per compute node


if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
  ################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
  redis_password=$(uuidgen)
  export redis_password
  echo "Redis password: ${redis_password}"

  nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
  nodes_array=( $nodes )

  node_1=${nodes_array[0]}
  ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address) # making redis-address
  port=6379
  ip_head=$ip:$port
  export ip_head
  echo "IP Head: $ip_head"

  echo "STARTING HEAD at $node_1"
  srun --nodes=1 --ntasks=1 -w $node_1 \
    ray start --head --node-ip-address="$node_1" --port=$port \
    --num-cpus $((SLURM_CPUS_PER_TASK)) --num-gpus $num_gpus --block &

  sleep 10

  worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
  for ((  i=1; i<=$worker_num; i++ ))
  do
    node_i=${nodes_array[$i]}
    echo "STARTING WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w $node_i \
      ray start --address "$node_1":"$port" \
      --num-cpus $((SLURM_CPUS_PER_TASK)) --num-gpus $num_gpus --block &
    sleep 5
  done

  echo All Ray workers started.
  ##############################################################################################
  # call your code below
fi


echo 'Starting training.'
# when training with Ray Train, --gpus should be equal to toal number of GPUs across the Ray Cluster
python3 -u mlpf/pyg_pipeline.py --train --ray-train \
    --config $1 \
    --prefix $2 \
    --ray-cpus $((SLURM_CPUS_PER_TASK*SLURM_JOB_NUM_NODES)) \
    --gpus $((SLURM_GPUS_PER_TASK*SLURM_JOB_NUM_NODES)) \
    --gpu-batch-multiplier 8 \
    --num-workers 8 \
    --prefetch-factor 16 \
    --experiments-dir /mnt/ceph/users/ewulff/particleflow/experiments \
    --local \
    --comet

echo 'Training done.'
