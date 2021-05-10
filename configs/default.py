# default FOCAL experiment settings
# all experiments should modify these settings only as needed
default_config = dict(
    env_name='cheetah-dir',
    n_train_tasks=2,
    n_eval_tasks=2,
    latent_size=20, # dimension of the latent context vector
    net_size=256, # number of units per FC layer in each network
    path_to_weights=None, # path to pre-trained weights to load into networks
    seed_list=[0, 1, 2], # list of random seeds
    env_params=dict(
        n_tasks=2, # number of distinct tasks in this domain, should equal sum of train and eval tasks
        randomize_tasks=True, # shuffle the tasks after creating them
        max_episode_steps=200, # built-in max episode length for this environment
    ),
    algo_params=dict(
        meta_batch=16, # number of tasks to average the gradient across
        batch_size=256, # number of transitions in the RL batch
        num_iterations=500, # number of data sampling / training iterates
        num_initial_steps=2000, # number of transitions collected per task before training
        num_tasks_sample=5, # number of randomly sampled tasks to collect data for each iteration
        num_steps_prior=400, # number of transitions to collect per task with z ~ prior
        num_steps_posterior=0, # number of transitions to collect per task with z ~ posterior
        num_extra_rl_steps_posterior=400, # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
        num_train_steps_per_itr=2000, # number of meta-gradient steps taken per iteration
        num_evals=2, # number of independent evals
        num_steps_per_eval=600,  # number of transitions to eval on        
        embedding_batch_size=64, # number of transitions in the context batch
        embedding_mini_batch_size=64, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        max_path_length=200, # max path length for this environment
        discount=0.99, # RL discount factor
        soft_target_tau=0.005, # for SAC target network update
        policy_lr=3e-4, # learning rate for policy
        qf_lr=3e-4, # learning rate for q network
        vf_lr=3e-4, # learning rate for v network
        context_lr=3e-4, # learning rate for context encoder
        c_lr=1e-4, # dual critic learning rate (BRAC dual)
        alpha_lr=1, # alpha learning rate (BRAC)
        c_iter=3, # number of dual critic steps per iteration
        policy_mean_reg_weight=1e-3, #
        policy_std_reg_weight=1e-3,
        policy_pre_activation_weight=0.,
        replay_buffer_size=1000000,
        save_replay_buffer=False,
        save_algorithm=False,
        save_environment=False,
        reward_scale=5., # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        sparse_rewards=False, # whether to sparsify rewards as determined in env
        kl_lambda=.1, # weight on KL divergence term in encoder loss
        use_information_bottleneck=False, # False makes latent context deterministic
        update_post_train=1, # how often to resample the context when collecting data during training (in trajectories)
        num_exp_traj_eval=1, # how many exploration trajs to collect before beginning posterior sampling at test time
        recurrent=False, # recurrent or permutation-invariant encoder
        dump_eval_paths=False, # whether to save evaluation trajectories
        sample=1, # whether to train with stochastic (noise-sampled) trajectories, for offline method (FOCAL) only
        train_epoch=6e5, # corresponding epoch of the model used to generate meta-training trajectories, offline method (FOCAL) only
        eval_epoch=6e5, # corresponding epoch of the model used to generate meta-testing trajectories, offline method (FOCAL) only
        divergence_name='kl', # divergence type in BRAC algo, offline method (FOCAL) only
        use_brac=True, # whether to use BRAC regularization (compare with batch PEARL)
        use_value_penalty=False, # whether to use value penalty in BRAC, only effective if use_brac=True
        train_alpha=True, # whether to train alpha (BRAC)
        alpha_init=500., # Initialized value for alpha (BRAC)
        alpha_max=2000., # Maximum value for alpha
        target_divergence=0.05, # For training alpha adaptively. As in BEAR, if train_alpha=True, increase alpha when div > target_divergence, lower alpha when div < target_divergence (BRAC)
        max_entropy=True, # whether to include max-entropy term (as in SAC and PEARL) in value function
        z_loss_weight=10, # z_loss weight
        use_next_obs_in_context=False, # use next obs if it is useful in distinguishing tasks
        allow_backward_z=False, # whether to allow gradients to flow back through z
        allow_eval=True, # if it is True, enable evaluation
        mb_replace=False, # meta batch sampling, replace or not
        dropout=0.1, # dropout for context encoder
        # data_dir="./data/walker_randparam_new_norm", # default data directory
        data_dir="./data/walker_randparam_new_norm", # default data directory
    ),
    util_params=dict(
        base_log_dir='output',
        use_gpu=True,
        gpu_id=0,
        debug=True, # debugging triggers printing and writes logs to debug directory
        docker=False, # TODO docker is not yet supported
        #machine='non_mujoco' #non_mujoco or mujoco, when non_mujoco is chosen, can train offline in non-mujoco environments
    ),
    algo_type='FOCAL'
)



