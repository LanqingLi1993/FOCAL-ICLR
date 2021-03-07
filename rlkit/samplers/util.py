import numpy as np
import logging

def offline_rollout(env, agent, buffer, max_path_length=np.inf, accum_context=True, animated=False, save_frames=False):
    # perform online rollout as in rollout()
    # If accum_context=True, aggregate offline context
    batch_dict = buffer.task_buffers[env._goal_idx].random_batch(max_path_length)
    # observations = batch_dict['observations']
    # actions = batch_dict['actions']
    # rewards = batch_dict['rewards']
    # terminals = batch_dict['terminals']
    # next_observations = batch_dict['next_observations']
    observations = []
    actions = []
    rewards = []
    terminals = []
    next_observations = []
    agent_infos = [{} for _ in actions]
    env_infos = [{} for _ in actions]
    path_length = 0

    if callable(getattr(env, "sparsify_rewards", None)):
        env_infos = [{'sparse_reward': env.sparsify_rewards(r)} for r in rewards]
    # batch_dict = dict(
    #     observations=self._observations[indices],
    #     actions=self._actions[indices],
    #     rewards=self._rewards[indices],
    #     terminals=self._terminals[indices],
    #     next_observations=self._next_obs[indices],
    #     sparse_rewards=self._sparse_rewards[indices],
    # )
    idx = 0


    if accum_context:
        # update the agent's current context
        # while idx < max_path_length:
        #     o = batch_dict['observations'][idx]
        #     a = batch_dict['actions'][idx]
        #     r = batch_dict['rewards'][idx]
        #     t = batch_dict['terminals'][idx]
        #     next_o = batch_dict['next_observations'][idx]
        #     if callable(getattr(env, "sparsify_rewards", None)):
        #         env_info = {'sparse_reward': env.sparsify_rewards(r)}
        #     else:
        #         env_info = {}
        #     print(o, a, r, o.shape, a.shape, r.shape)
        #     agent.update_context([o, a, r, next_o, t, env_info])
        #     idx += 1
        agent.update_context_dict(batch_dict=batch_dict, env=env)

    agent.infer_posterior(agent.context, task_indices=env._goal_idx)

    o = env.reset()
    # perform online evaluation
    while path_length < max_path_length:
        logging.info('task_indices:')
        logging.info(agent.task_indices)
        logging.info('z')
        logging.info(agent.z)
        logging.info('o:')
        logging.info(o)
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        logging.info('next_o:')
        logging.info(next_o)
        logging.info('a')
        logging.info(a)
        logging.info('r')
        logging.info(r)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o
        if animated:
            env.render()
        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image
        env_infos.append(env_info)
        if d:
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
    next_observations = np.array(next_observations)
    if len(next_observations.shape) == 1:
        next_observations = np.expand_dims(next_observations, 1)

    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def offline_sample(env, agent, buffer, max_path_length=np.inf, accum_context=True):
    """
       The following value for the following keys will be a 2D array, with the
       first dimension corresponding to the time dimension.
        - observations
        - actions
        - rewards
        - next_observations
        - terminals

       The next two elements will be lists of dictionaries, with the index into
       the list being the index into the time
        - agent_infos
        - env_infos

       :param env:
       :param agent:
       :param max_path_length:
       :param accum_context: if True, accumulate the collected context
       :param animated:
       :param save_frames: if True, save video of rollout
       :return:
       """
    # observations = []
    # actions = []
    # rewards = []
    # terminals = []

    # o = env.reset()
    # goal_idx = env.goal_idx
    # next_o = None
    # path_length = 0

    batch_dict = buffer.task_buffers[env._goal_idx].random_batch(max_path_length)
    observations = batch_dict['observations']
    actions = batch_dict['actions']
    rewards = batch_dict['rewards']
    terminals = batch_dict['terminals']
    next_observations = batch_dict['next_observations']
    agent_infos = [{} for _ in actions]
    env_infos = [{} for _ in actions]

    if callable(getattr(env, "sparsify_rewards", None)):
        env_infos = [{'sparse_reward': env.sparsify_rewards(r)} for r in rewards]
    # batch_dict = dict(
    #     observations=self._observations[indices],
    #     actions=self._actions[indices],
    #     rewards=self._rewards[indices],
    #     terminals=self._terminals[indices],
    #     next_observations=self._next_obs[indices],
    #     sparse_rewards=self._sparse_rewards[indices],
    # )

    idx = 0
    if accum_context:
        # while idx < max_path_length:
        #     o = batch_dict['observations'][idx]
        #     a = batch_dict['actions'][idx]
        #     r = batch_dict['rewards'][idx]
        #     t = batch_dict['terminals'][idx]
        #     next_o = batch_dict['next_observations'][idx]
        #     if callable(getattr(env, "sparsify_rewards", None)):
        #         env_info = {'sparse_reward': env.sparsify_rewards(r)}
        #     print(o, a, r, o.shape, a.shape, r.shape)
        #     agent.update_context([o, a, r, next_o, t, env_info])
        #     idx += 1
        agent.update_context_dict(batch_dict=batch_dict, env=env)
        # agent.infer_posterior(agent.context, task_indices=np.array([env._goal_idx]))

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
    next_observations = np.array(next_observations)
    if len(next_observations.shape) == 1:
        next_observations = np.expand_dims(next_observations, 1)


    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )



def rollout(env, agent, max_path_length=np.inf, accum_context=True, animated=False, save_frames=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated:
    :param save_frames: if True, save video of rollout
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0


    if animated:
        env.render()
    while path_length < max_path_length:
        logging.info('task_indices:')
        logging.info(agent.task_indices)
        logging.info('z')
        logging.info(agent.z)
        logging.info('o:')
        logging.info(o)
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        logging.info('next_o:')
        logging.info(next_o)
        logging.info('a')
        logging.info(a)
        logging.info('r')
        logging.info(r)

        # update the agent's current context
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o
        if animated:
            env.render()
        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image
        env_infos.append(env_info)
        if d:
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
