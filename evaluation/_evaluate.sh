#!/usr/bin/env bash

# custom config

# example:
# bash ./evaluation/_evaluate.sh stylegan2-ffhq-config-f Photo2Disney_gt100 Photo Disney NADA

Src_Gen=$1 # stylegan2-ffhq-config-f
Tar_Gen=$2 # Photo2Disney_gt100
Src_Cls=$3 # Photo
Tar_Cls=$4 # Disney
Method=$5 # NADA/IPL

output_path=${Tar_Gen}_${Method}
echo ${output_path}

echo "computing FID"
# FID
if [ -d ./eval_FIDnISnLPIPS/${output_path} ]; then
  echo "Oops! The results exist at ./eval_FIDnISnLPIPS/${output_path} (so skip this job)"
else
  CUDA_VISIBLE_DEVICES=0 python ./evaluation/generate.py --ckpt_target ./output_stylegan/${Method}/${Tar_Gen}.pt \
  --source ${Src_Cls} --target ${Tar_Cls} --mode eval_FIDnISnLPIPS --FIDnISnLPIPS_sample 5000 \
#  --size 256
fi
CUDA_VISIBLE_DEVICES=0 python ./evaluation/fid_score.py --path ./eval_FIDnISnLPIPS/${output_path}/images /home/paperspace/Desktop/dataset/stargan-v2/data/train/${Tar_Cls} --device cuda

echo "computing IS"
# IS
if [ -d ./eval_FIDnISnLPIPS/${output_path} ]; then
  echo "Oops! The results exist at ./eval_FIDnISnLPIPS/${output_path} (so skip this job)"
else
  CUDA_VISIBLE_DEVICES=0 python ./evaluation/generate.py --ckpt_target ./output_stylegan/${Method}/${Tar_Gen}.pt \
  --source ${Src_Cls} --target ${Tar_Cls} --mode eval_FIDnISnLPIPS --FIDnISnLPIPS_sample 1000 \
#  --size 256
fi
CUDA_VISIBLE_DEVICES=0 python ./evaluation/inception_score.py --save_target ./eval_FIDnISnLPIPS/${output_path} --IS_sample 1000

echo "computing LPIPS"
# Intra-LPIPS (In 1-shot setting, compute LPIPS of all generated images to 1 image)
if [ -d ./eval_FIDnISnLPIPS/${output_path} ]; then
  echo "Oops! The results exist at ./eval_FIDnISnLPIPS/${output_path} (so skip this job)"
else
  CUDA_VISIBLE_DEVICES=0 python ./evaluation/generate.py --ckpt_target ./output_stylegan/${Method}/${Tar_Gen}.pt \
  --source ${Src_Cls} --target ${Tar_Cls} --mode eval_FIDnISnLPIPS --FIDnISnLPIPS_sample 1000 \
#  --size 256
fi
#CUDA_VISIBLE_DEVICES=0 python ./evaluation/IntraLPIPS.py --img_pth ./eval_FIDnISnLPIPS/${output_path} --LPIPS_sample 1000 --target_cl dog
CUDA_VISIBLE_DEVICES=0 python ./evaluation/IntraLPIPS.py --intra_lpips_path ./eval_FIDnISnLPIPS/${output_path} --data_path

#rm -R ./eval_FIDnISnLPIPS/${output_path}

echo "computing SCS"
# SCS
if [ -d ./eval_SCS/${output_path} ]; then
  echo "Oops! The results exist at ./eval_SCS/${output_path} (so skip this job)"
else
  CUDA_VISIBLE_DEVICES=0 python ./evaluation/generate.py --ckpt_source ./pretrain/${Src_Gen}.pt \--ckpt_target ./output_stylegan/${Method}/${Tar_Gen}.pt \
  --source ${Src_Cls} --target ${Tar_Cls} --mode eval_SCS --SCS_samples 500 \
#  --size 256
fi
CUDA_VISIBLE_DEVICES=0 python ./evaluation/scs_score.py --img_pth ./eval_SCS/${output_path}


#rm -R ./eval_SCS/${output_path}

#SIFID
echo "computing SIFID"
if [ -d ./eval_SIFID/${output_path} ]; then
  echo "Oops! The results exist at ./eval_SIFID/${output_path} (so skip this job)"
else
  CUDA_VISIBLE_DEVICES=0 python ./evaluation/generate.py --ckpt_target ./output_stylegan/${Method}/${Tar_Gen}.pt \
  --source ${Src_Cls} --target ${Tar_Cls} --mode eval_SIFID --SIFID_sample 1000 \
#  --size 256
fi

for R in 1 2 3
do
  CUDA_VISIBLE_DEVICES=0 python ./evaluation/sifid_score.py --path2real ./eval_SIFID/${Tar_Cls}/R${R} --path2fake ./eval_SIFID/${output_path}/images
done
