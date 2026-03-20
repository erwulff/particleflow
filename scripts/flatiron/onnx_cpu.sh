#!/bin/sh

# Walltime limit
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p genx
#SBATCH --cpus-per-task=64

# Job name
#SBATCH -J onnx_cpu

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err


module --force purge; module load modules/2.4-20250724
module load slurm gcc cmake cuda/12.8.0 cudnn/9.2.0.82-12 nccl openmpi apptainer

nvidia-smi
export PYTHONPATH=`pwd`
source ~/miniforge3/bin/activate mlpf
which python3
python3 --version


python scripts/cms-validate-onnx.py \
  --checkpoint experiments/after_fix_pyg-cms-v1_cms_run3_20260319_171021_176913/checkpoints/checkpoint-10000.pth \
  --model-kwargs experiments/after_fix_pyg-cms-v1_cms_run3_20260319_171021_176913/model_kwargs.pkl \
  --data-dir /mnt/ceph/users/ewulff/tensorflow_datasets/cms \
  --dataset cms_pf_ttbar \
  --num-events 500 \
  --device cpu \
  --num-threads-torch-intra 64 \
  --num-threads-torch-inter 1 \
  --num-threads-onnx-intra 64 \
  --num-threads-onnx-inter 1 \
  --outdir onnx_benchmarks/cpu
