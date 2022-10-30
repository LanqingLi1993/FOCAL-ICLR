# FOCAL: Efficient Fully-Offline Meta-Reinforcement Learning Via Distance Metric Learning and Behavior Regularization
<!-- 
> Meta-learning for offline reinforcement learning (OMRL) is an understudied problem with tremendous potential impact by enabling RL algorithms in many real-world applications. A popular solution to the problem is to infer task identity as augmented state using a context-based encoder, for which efficient learning of task representations remains an open challenge. In this work, we improve upon one of the SOTA OMRL algorithms, FOCAL, by incorporating intra-task attention mechanism and inter-task contrastive learning objectives for more effective task inference and learning of control. Theoretical analysis and experiments are presented to demonstrate the superior performance, efficiency and robustness of our end-to-end and model-free method compared to prior algorithms across multiple meta-RL benchmarks. -->

> We study the offline meta-reinforcement learning (OMRL) problem, a paradigm which enables reinforcement learning (RL) algorithms to quickly adapt to unseen tasks without any interactions with the environments, making RL truly practical in many real-world applications. This problem is still not fully understood, for which two major challenges need to be addressed. First, offline RL usually suffers from bootstrapping errors of out-of-distribution state-actions which leads to divergence of value functions. Second, meta-RL requires efficient and robust task inference learned jointly with control policy. In this work, we enforce behavior regularization on learned policy as a general approach to offline RL, combined with a deterministic context encoder for efficient task inference. We propose a novel negative-power distance metric on bounded context embedding space, whose gradients propagation is detached from the Bellman backup. We provide analysis and insight showing that some simple design choices can yield substantial improvements over recent approaches involving meta-RL and distance metric learning. To the best of our knowledge, our method is the first model-free and end-to-end OMRL algorithm, which is computationally efficient and demonstrated to outperform prior algorithms on several meta-RL benchmarks.

## Installation
To install locally, you will need to first install [MuJoCo](https://www.roboti.us/index.html). For task distributions in which the reward function varies (Cheetah, Ant), install MuJoCo150 or plus. Set `LD_LIBRARY_PATH` to point to both the MuJoCo binaries (`/$HOME/.mujoco/mujoco200/bin`) as well as the gpu drivers (something like `/usr/lib/nvidia-390`, you can find your version by running `nvidia-smi`).

For the remaining dependencies, create conda environment by
```
conda env create -f environment.yaml
```

<!-- For task distributions where the transition function (dynamics)  varies  -->

**For Walker environments**, MuJoCo131 is required.
Simply install it the same way as MuJoCo200. To swtch between different MuJoCo versions:

```
export MUJOCO_PY_MJPRO_PATH=~/.mujoco/mjpro${VERSION_NUM}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro${VERSION_NUM}/bin
``` 

The environments make use of the module `rand_param_envs` which is submoduled in this repository. Add the module to your python path, `export PYTHONPATH=./rand_param_envs:$PYTHONPATH` (Check out [direnv](https://direnv.net/) for handy directory-dependent path managenement.)


This installation has been tested only on 64-bit CentOS 7.2. The whole pipeline consists of two stages: **data generation** and **Offline RL experiments**:

## Data Generation

FOCAL requires fixed data (batch) for meta-training and meta-testing, which are generated by trained [SAC](https://arxiv.org/pdf/1801.01290.pdf) behavior policies. Experiments at this stage are configured via `train.yaml` located in `./rlkit/torch/sac/pytorch_sac/config/`.  

Example of training policies and generating trajectories on multiple tasks:

```
python policy_train.py --gpu 0
```

Generate trajectories from pretrained models

```
python policy_train.py --eval
```

Generated data will be saved in `./data/`

## Offline RL Experiments
Experiments are configured via `json` configuration files located in `./configs`. Basic settings are defined and described in `./configs/default.py`. To reproduce an experiment, run: 
```
python launch_experiment.py ./configs/[EXP].json
```
By default the code will use the GPU - to use CPU instead, set `use_gpu=False` in the corresponding config file.

Output files will be written to `./output/[ENV]/[EXP NAME]` where the experiment name corresponds to the process starting time. The file `progress.csv` contains statistics logged over the course of training. `data_epoch_[EPOCH].csv` contains embedding vector statistics. We recommend `viskit` for visualizing learning curves: https://github.com/vitchyr/viskit. Network weights are also snapshotted during training.

To evaluate a learned policy after training has concluded, run `sim_policy.py`. This script will run a given policy across a set of evaluation tasks and optionally generate a video of these trajectories. Rendering is offline and the video is saved to the experiment folder.

Example of running experiment on walker_rand_params environment:


- download [walker data](https://drive.google.com/file/d/1zdaUX-LC8c6AaS9We85bUvoA_9iHZcyg/view?usp=sharing) and unzip the data to `./data/walker_rand_params` (Download all normalized offline training data used in the FOCAL paper [here](https://share.weiyun.com/5kqk9s7S))
- edit walker_rand_params.json to add dump_eval_paths=1 and data_dir=`./data/walker_rand_params`
- run python launch_experiment.py ./configs/walker_rand_params.json

## Reproducing Result in [FOCAL Paper](https://openreview.net/forum?id=8cpHIfgY4Dj)

We provide code for reproducing figure 2-9 and table 1 in generate_plot.py. Use [output data](https://drive.google.com/file/d/1ZOF68UHCVAHPPEJBbYutfXpBD567J20U/view?usp=sharing) to download the output files required for visualization and add them to `./output/` directory. To produce all figures at a time, run
```
python3 generate_plot.py
```

To produce each figure individually, run the function named by the corresponding figure number in main().

## References

```
@inproceedings{li2021focal,
  title={{FOCAL}: Efficient Fully-Offline Meta-Reinforcement Learning via Distance Metric Learning and Behavior Regularization},
  author={Lanqing Li and Rui Yang and Dijun Luo},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=8cpHIfgY4Dj}
}
```


