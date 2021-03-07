import numpy as np
from gym import spaces
from gym import Env

#from rlkit.envs.mujoco.half_cheetah import HalfCheetahEnv
from rlkit.envs import register_env


@register_env('cheetah-vel')
class HalfCheetahVelEnv(Env):

    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, max_episode_steps=200):
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal_vel = self.tasks[0].get('velocity', 0.0)
        self._goal = self._goal_vel

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,))
        self._max_episode_steps = max_episode_steps
        self._step = 0


    def step(self, action):
        print("placeholder! no env no step")
        pass

    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        velocities = np.random.uniform(0.0, 3.0, size=(num_tasks,))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))
    
    def reset(self):
        return self.reset_model()

    def reset_model(self):
        # reset to a random location on the unit square
        self._state = np.random.uniform(-1., 1., size=(20,))
        self._step = 0
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self._state)

    def reset_task(self, idx):
        self._goal_idx = idx
        self._task = self.tasks[idx]
        self._goal_vel = self._task['velocity']
        self._goal = self._goal_vel
        self.reset()
    
    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('no render')
        pass
