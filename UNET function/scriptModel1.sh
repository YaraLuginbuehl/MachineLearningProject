#!/bin/sh

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=16000
#SBATCH --tmp=4000                     
#SBATCH --job-name=cnn_yaralug
#SBATCH --output=cnn_moon.out
#SBATCH --error=cnn_moon.err

cd /cluster/home/yluginbuehl

export PYTHONPATH=/cluster/home/yluginbuehl:$PYTHONPATH

/cluster/home/yluginbuehl/MLenv2/bin/python -m Denoising_2D.UNET.Model1.CNN_YaraLug_UNET_Model1  > /cluster/home/yluginbuehl/Denoising_2D/UNET/Model1/Model1.txt 2>&1
