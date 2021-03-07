import numpy as np
from gym import spaces
from gym import Env

#from rlkit.envs.mujoco.ant_multitask_base import MultitaskAntEnv
from rlkit.envs import register_env


@register_env('ant-dir')
class AntDirEnv(Env):

    def __init__(self, task={}, n_tasks=2, forward_backward=False, max_episode_steps=200, randomize_tasks=True, **kwargs):
        self.forward_backward = forward_backward
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal = self.tasks[0]['goal']

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(27,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,))
        self._max_episode_steps = max_episode_steps
        self._step = 0

    def step(self, action):
        print("placeholder! no env no step")
        pass

    def sample_tasks(self, num_tasks):
        if self.forward_backward:
            assert num_tasks == 2
            velocities = np.array([0., np.pi])
        else:
            velocities = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        tasks = [{'goal': velocity} for velocity in velocities]
        return tasks
    
    def get_all_task_idx(self):
        return range(len(self.tasks))
    
    def reset(self):
        return self.reset_model()

    def reset_model(self):
        # reset to a random location on the unit square
        self._state = np.random.uniform(-1., 1., size=(27,))
        self._step = 0
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self._state)

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector
        self._goal_idx = idx
        self.reset()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('no render')
        pass
