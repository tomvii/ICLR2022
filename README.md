# ICLR2022

## Update 12/06/2021 : tieredImagenet Results
We trained a WRN-28-10 backbone on tieredImageNet dataset which contains 608 classes sampled from hierarchical categories. Each class belongs to one of 34 higher- level categories sampled from the high-level nodes in the ImageNet. We use 351, 97, and 160 classes for training, validation, and test, respectively. Before training, we downsampled the images to 32x32 due to resource constraints, because of which our accuracies are not directly comparable to the ones reported in DC paper (Yang et. al). Regardless, DC+ shows a consistent improvement over DC as expected.

| Method      	  | 5way-1shot      	| 5way-5shot      	|
|--------------	  |-----------------	|-----------------	|
| DC 	            | 66.899 +- 0.607 	| 84.074 +- 0.434 	|
| DC+          	  | 69.350 +- 0.587 	| 85.236 +- 0.416	  |


## Setup environment
```conda create -n myenv python=3.6```

```conda activate myenv```

Get torch

```pip install --user torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html```

If torch==1.7.1+cu101 fails, install torch==1.7.1 instead,

```pip install --user torch==1.7.1 -f https://download.pytorch.org/whl/torch_stable.html```

Get remaining packages

```pip install optuna==2.10.0```

```pip install matplotlib==3.3.4```


## Get the features
You can train your own feature extractor following https://github.com/nupurkmr9/S2M2_fewshot

Or you can download miniImagenet, CUB extracted features from https://drive.google.com/drive/folders/1IjqOYLRH0OwkMZo8Tp4EG02ltDppi61n

Put the extracted features in ICLR2022/data/miniImagenet or ICLR2022/data/CUB directories

Create a directory ICLR2022/logs to keep track of experiments

## Train our proposed method
Use the following script to replicate our results from the paper,

1. miniImagenet

5 way 1 shot
```python evaluate_DC_optuna.py --n_shot 1 --n_ways 5 --dataset miniImagenet --no_optuna --n_runs 5000 --print_iter 200 --m 1 --k 8 --alpha 3000 --alpha2 10 --beta 0.5 --use_dc_from_scaling_v4_cc5```

5 way 5 shot
```python evaluate_DC_optuna.py --n_shot 5 --n_ways 5 --dataset miniImagenet --no_optuna --n_runs 5000 --print_iter 200 --m 3 --k 30 --alpha 9000 --alpha2 10 --beta 0.5 --use_dc_from_scaling_v4_cc5```

2. CUB

5 way 1 shot
```python evaluate_DC_optuna.py --n_shot 1 --n_ways 5 --dataset CUB --no_optuna --n_runs 5000 --print_iter 200 --m 1 --k 4 --alpha 8000 --alpha2 10 --beta 0.5 --use_dc_from_scaling_v4_cc5```

5 way 5 shot
```python evaluate_DC_optuna.py --n_shot 5 --n_ways 5 --dataset CUB --no_optuna --n_runs 5000 --print_iter 200 --m 2 --k 4 --alpha 5000 --alpha2 10 --beta 0.5 --use_dc_from_scaling_v4_cc5```


## Results
| dataset      	| 5way-1shot      	| 5way-5shot      	|
|--------------	|-----------------	|-----------------	|
| miniImagenet 	| 73.006 +- 0.501 	| 87.226 +- 0.331 	|
| CUB          	| 84.574 +- 0.489 	| 93.466 +- 0.250 	|


