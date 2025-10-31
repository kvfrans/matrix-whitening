from logging import config
import random
import numpy as np

job_list = []
debug_config = '--model.hidden_size 64 --model.depth 2 --model.num_heads 2 --model.mlp_ratio 1'
tiny_config = '--model.hidden_size 128 --model.depth 4 --model.num_heads 4 --model.mlp_ratio 4' 
xxsmall_config = '--model.hidden_size 72 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
xsmall_config = '--model.hidden_size 144 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
small_config = '--model.hidden_size 384 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
big_config = '--model.hidden_size 768 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
xlarge_config = '--model.hidden_size 1152 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'

dit_config = '--model.train_type dit --dataset_name imagenet256 --fid_stats data/imagenet256_fidstats_ours.npz --model.cfg_scale 1.5 --model.class_dropout_prob 0.1 --model.patch_size 2 --model.lr_adam_input 0.0001 --model.lr_adam_output 0.001 --model.lr_adam_layernorm 0.0001 --batch_size 1024'
vit_config = '--model.train_type vit --dataset_name imagenet256-augment --model.patch_size 16 --model.use_stable_vae 0 --model.lr_adam_input 0.0001 --model.lr_adam_output 0.001 --model.lr_adam_layernorm 0.0001 --batch_size 256'
gpt_config = '--model.train_type gpt --dataset_name openwebtext --model.use_stable_vae 0 --model.lr_adam_input 0.01 --model.lr_adam_output 0.01 --model.lr_adam_layernorm 0.01'


from best_params import best_params
def get_vals(optimizer):
    return best_params[optimizer]['lr'], best_params[optimizer]['weight_decay'], best_params[optimizer]['b1'], best_params[optimizer]['b2'], best_params[optimizer]['command']
sqrt_10 = 3.162
sqrt_sqrt_10 = 1.778
sqrt_sqrt_sqrt_10 = 1.333

# This will just run the baseline, optimal hyperparameters.
base = f'python train.py --wandb.group Baseline --model.sharding fsdp --log_interval 100 --max_steps 10_000 --model.warmup 200 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4 --model.sequence_length 256 --model.hidden_size 768'
for use_cosine in [1]:
    for name, config in [('GPT', gpt_config)]:
        for optimizer in ['adam', 'muon', 'adamuon', 'soap100', 'splus100', 'psgd', 'signum']:
            lr, wd, b1, b2, command = get_vals(optimizer)
            job_list.append(base + f'  --wandb.name {name}-Baseline-LR{lr}-WD{wd}-M{b1}-V{b2} --label {optimizer} --model.lr {lr} {command} {config} --model.use_cosine_decay {use_cosine} --model.weight_decay {wd} --model.beta1 {b1} --model.beta2 {b2}')

# This will recreate the graph in Fig 2, sanity checking that all hyperparameters are optimal (i.e. at a local minimum.)
base = f'python train.py --wandb.group HyperparameterSweep --model.sharding fsdp --log_interval 100 --max_steps 10_000 --model.warmup 200 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4 --model.sequence_length 256 --model.hidden_size 768'
for use_cosine in [1]:
    for name, config in [('GPT', gpt_config)]:
        for optimizer in ['adam', 'muon', 'adamuon', 'soap100', 'splus100', 'psgd', 'signum']:

            lr, wd, b1, b2, command = get_vals(optimizer)
            job_list.append(base + f'  --wandb.name {name}-Baseline-LR{lr}-WD{wd}-M{b1}-V{b2} --label {optimizer} --model.lr {lr} {command} {config} --model.use_cosine_decay {use_cosine} --model.weight_decay {wd} --model.beta1 {b1} --model.beta2 {b2}')

            lr_base, wd, b1, b2, command = get_vals(optimizer)
            for lr in [lr_base / sqrt_10, lr_base / sqrt_sqrt_10, lr_base / sqrt_sqrt_sqrt_10, lr_base * sqrt_sqrt_sqrt_10, lr_base * sqrt_sqrt_10, lr_base * sqrt_10]:
                job_list.append(base + f'  --wandb.name {name}-LR{lr} --label {optimizer} --model.lr {lr} {command} {config} --model.use_cosine_decay {use_cosine} --model.weight_decay {wd} --model.beta1 {b1} --model.beta2 {b2}')

            lr, wd_base, b1, b2, command = get_vals(optimizer)
            for wd in [wd_base / sqrt_10, wd_base / sqrt_sqrt_10, wd_base * sqrt_sqrt_10, wd_base * sqrt_10]:
                job_list.append(base + f'  --wandb.name {name}-WD{wd} --label {optimizer} --model.lr {lr} {command} {config} --model.use_cosine_decay {use_cosine} --model.weight_decay {wd} --model.beta1 {b1} --model.beta2 {b2}')

            lr, wd, b1_base, b2, command = get_vals(optimizer)
            b1_base_flip = 1-b1_base
            for b1_flip in [b1_base_flip / sqrt_sqrt_10, b1_base_flip * sqrt_sqrt_10]:
                b1 = 1 - b1_flip
                job_list.append(base + f'  --wandb.name {name}-M{b1} --label {optimizer} --model.lr {lr} {command} {config} --model.use_cosine_decay {use_cosine} --model.weight_decay {wd} --model.beta1 {b1} --model.beta2 {b2}')

            lr, wd, b1, b2_base, command = get_vals(optimizer)
            if b2_base != 0:
                b2_base_flip = 1-b2_base
                for b2_flip in [b2_base_flip / sqrt_10, b2_base_flip * sqrt_10]:
                    b2 = 1 - b2_flip
                    job_list.append(base + f'  --wandb.name {name}-V{b2} --label {optimizer} --model.lr {lr} {command} {config} --model.use_cosine_decay {use_cosine} --model.weight_decay {wd} --model.beta1 {b1} --model.beta2 {b2}')

for job in job_list:
    pass
    # Run the commands on your cluster of choosing here.
    # queue_job(job)