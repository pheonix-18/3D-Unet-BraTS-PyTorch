#!/bin/bash
#SBATCH --job-name=3DU
#SBATCH -p gpu
#SBATCH --gres=gpu:volta:1

#parallel details
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem-per-cpu=6144M


echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

nvidia-smi

# Load the TensorFlow module
module load anaconda3
module list

source activate pytorch

#which python

# export PYTHONPATH=$PYTHONPATH:/home/tu666280/Deformable-DETR

# echo "Environment Activated"


# python main.py --epochs 400 --lr_drop 200 --batch_size 3 --resume pretrained_weights/r50_deformable_detr-checkpoint.pth --dataset_file nps --output_dir nps_r50_dedetr_more_epochs --wandb
#
python3 train.py
# You're done!
echo "Ending script..."%                            
