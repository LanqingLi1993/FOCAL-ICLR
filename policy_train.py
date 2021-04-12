"""
Training behavior policies for FOCAL

"""

import click
import json
import os
from hydra.experimental import compose, initialize

import argparse
import multiprocessing as mp
from multiprocessing import Pool
from itertools import product

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs import ENVS
from configs.default import default_config


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

initialize(config_dir="rlkit/torch/sac/pytorch_sac/config/")
cfg = compose("train.yaml")
def experiment(variant, cfg=cfg, goal_idx=0, seed=0,  eval=False):
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    if seed is not None:
        env.seed(seed) 
    env.reset_task(goal_idx)
    if "cuda" in cfg.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    # NOTE: for new environment variable to be effective, torch should be imported after assignment
    from rlkit.torch.sac.pytorch_sac.train import Workspace
    workspace = Workspace(cfg=cfg, env=env, env_name=variant['env_name'], goal_idx=goal_idx)
    if eval:
        print('evaluate:')
        workspace.run_evaluate()
    else:
        workspace.run()


@click.command()
@click.argument("config", default="./configs/sparse-point-robot.json")
@click.option("--gpu", default=0)
@click.option("--docker", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
@click.option("--eval", is_flag=True, default=False)
def main(config, gpu, docker, debug, eval, goal_idx=0, seed=0):
    variant = default_config
    cwd = os.getcwd()
    files = os.listdir(cwd)
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu

    cfg.gpu_id = gpu
    print('cfg.agent', cfg.agent)
    print(list(range(variant['env_params']['n_tasks'])))
    # multi-processing
    p = mp.Pool(mp.cpu_count())
    if variant['env_params']['n_tasks'] > 1:
        p.starmap(experiment, product([variant], [cfg], list(range(variant['env_params']['n_tasks']))))
    else:
        experiment(variant=variant, cfg=cfg, goal_idx=goal_idx)


if __name__ == '__main__':
    #add a change 
    main()
