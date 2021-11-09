# ICLR2022

## Get the features
You can train your own feature extractor following https://github.com/nupurkmr9/S2M2_fewshot

Or you can download miniImagenet, CUB extracted features from https://drive.google.com/drive/folders/1IjqOYLRH0OwkMZo8Tp4EG02ltDppi61n

Put the extracted features in ICLR2022/data/miniImagenet or ICLR2022/data/CUB directories

Create a directory ICLR2022/logs to keep track of experiments

## Train our proposed method
Use the following script to replicate our results from the paper,

1. miniImagenet

5 way 1 shot
```python ICLR2022/evaluate_DC_optuna.py --n_shot 1 --n_ways 5 --dataset miniImagenet --no_optuna --n_runs 5000 --print_iter 200 --m 1 --k 8 --alpha 3000 --alpha2 10 --beta 0.5 --use_dc_from_scaling_v4_cc5```

5 way 5 shot
```python ICLR2022/evaluate_DC_optuna.py --n_shot 5 --n_ways 5 --dataset miniImagenet --no_optuna --n_runs 5000 --print_iter 200 --m 3 --k 30 --alpha 9000 --alpha2 10 --beta 0.5 --use_dc_from_scaling_v4_cc5```

2. CUB

5 way 1 shot
```python ICLR2022/evaluate_DC_optuna.py --n_shot 1 --n_ways 5 --dataset CUB --no_optuna --n_runs 5000 --print_iter 200 --m 1 --k 4 --alpha 8000 --alpha2 10 --beta 0.5 --use_dc_from_scaling_v4_cc5```

5 way 5 shot
```python ICLR2022/evaluate_DC_optuna.py --n_shot 5 --n_ways 5 --dataset CUB --no_optuna --n_runs 5000 --print_iter 200 --m 2 --k 4 --alpha 5000 --alpha2 10 --beta 0.5 --use_dc_from_scaling_v4_cc5```

