"""
Launcher for experiments with PEARL

"""

import click
import json
import os
from hydra.experimental import compose, initialize
import argparse
import multiprocessing as mp
from multiprocessing import Pool

from rlkit.torch.sac.pytorch_sac.train import Workspace
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


# @hydra.main(config_path='rlkit/torch/sac/pytorch_sac/config/train.yaml', strict=True)
# c = []
# hydra.main(config_path='rlkit/torch/sac/pytorch_sac/config/train.yaml')(c.append)()
# print(c)
# cfg = c[0]
# print(cfg)
initialize(config_dir="rlkit/torch/sac/pytorch_sac/config/")
cfg = compose("train.yaml")
print(cfg.agent)
def experiment(cfg=cfg, env=None, goal_idx=0):
    workspace = Workspace(cfg=cfg, env=env, goal_idx=goal_idx)
    workspace.run_evaluate()


# @click.command()
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="./configs/sparse-point-robot.json")
args = parser.parse_args()
# @click.argument('config', default=None)
# @click.option('--gpu', default=0)
# @click.option('--docker', is_flag=True, default=False)
# @click.option('--debug', is_flag=True, default=False)
def main(goal_idx=0, args=args):
    variant = default_config
    if args.config:
        with open(os.path.join(args.config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    # variant['util_params']['gpu_id'] = gpu
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    env.reset_task(goal_idx)
    experiment(env=env, goal_idx=goal_idx)

if __name__ == '__main__':
    variant = default_config
    if args.config:
        with open(os.path.join(args.config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    with Pool(mp.cpu_count()) as p:
        p.map(main, list(range(variant['env_params']['n_tasks'])))
