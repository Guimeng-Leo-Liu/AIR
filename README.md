# AIR

## Setup

The code relies on the official implementation of [CLIP](https://github.com/openai/CLIP), 
and the [Rosinality](https://github.com/rosinality/stylegan2-pytorch/) pytorch implementation of StyleGAN2.

In addition, run the following commands:
  ```shell script
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
pip install ftfy regex tqdm tensorboard tensorboardx
pip install git+https://github.com/openai/CLIP.git
```

To convert a generator from one domain to another, run the training script:

```
bash adaptation.sh
```
