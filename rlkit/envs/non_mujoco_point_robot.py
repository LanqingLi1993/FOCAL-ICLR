import numpy as np
from gym import spaces
from gym import Env

from . import register_env


@register_env('point-robot')
class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, randomize_tasks=False, n_tasks=2, max_episode_steps=200):

        if randomize_tasks:
            np.random.seed(1337)
            goals = [[np.random.uniform(-1., 1.), np.random.uniform(-1., 1.)] for _ in range(n_tasks)]
        else:
            # some hand-coded goals for debugging
            goals = [np.array([10, -10]),
                     np.array([10, 10]),
                     np.array([-10, 10]),
                     np.array([-10, -10]),
                     np.array([0, 0]),

                     np.array([7, 2]),
                     np.array([0, 4]),
                     np.array([-6, 9])
                     ]
            goals = [g / 10. for g in goals]
        self.goals = goals

        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))
        self._max_episode_steps = max_episode_steps
        self._step = 0

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal_idx = idx
        self._goal = self.goals[idx]
        self.reset()

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_model(self):
        # reset to a random location on the unit square
        self._state = np.random.uniform(-1., 1., size=(2,))
        self._step = 0
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = False
        ob = self._get_obs()
        self._step += 1
        if self._step >= self._max_episode_steps:
            done = True
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)


@register_env('point-robot-wind')
class PointWindEnv(PointEnv):
    '''
     - goal is fixed but unknown on unit half circle
     - a positional shift is applied at every step, which is sampled for every task uniformly within [-0.05, 0.05]^2
     - reward is L2 distance given only within goal radius

     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=False, n_tasks=2, max_episode_steps=200, goal_radius=0.2, goal_idx=None):
        #super().__init__(randomize_tasks, n_tasks, max_episode_steps)
        self.goal_radius = goal_radius

        # if randomize_tasks:
        np.random.seed(1337)
        radius = 1.0
        angles = np.linspace(0, np.pi, num=n_tasks)
        #xs = radius * np.cos(angles)
        #ys = radius * np.sin(angles)
        xs = np.ones((n_tasks, ))
        ys = np.zeros((n_tasks, ))
        goals = np.stack([xs, ys], axis=1)
        wind_x = np.random.uniform(-0.05, 0.05, n_tasks) 
        wind_y = np.random.uniform(-0.05, 0.05, n_tasks) 
        winds = np.stack([wind_x, wind_y], axis=1)
        # if randomize_tasks:
            # np.random.shuffle(goals)
        goals = goals.tolist()
        self.winds = winds
        
        super().__init__(randomize_tasks, n_tasks, max_episode_steps)
        self.goals = goals
        if isinstance(goal_idx, int):
            self.reset_task(goal_idx)
        else:
            self.reset_task(0)

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal_idx = idx
        self._goal = self.goals[idx]
        self._wind = self.winds[idx]
        self.reset()

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        #mask = (r >= -self.goal_radius).astype(np.float32)
        mask = (r >= -self.goal_radius)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        self._step = 0
        return self._get_obs()

    def step(self, action):
        # print('goal', self._goal)
        self._state = self._state + action + self._wind
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = False
        ob = self._get_obs()
        self._step += 1
        if self._step >= self._max_episode_steps:
            done = True
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d = dict()
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d
        # return ob, sparse_reward, done, d

@register_env('sparse-point-robot')
class SparsePointEnv(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius

     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=False, n_tasks=2, max_episode_steps=200, goal_radius=0.05, goal_idx=None):
        super().__init__(randomize_tasks, n_tasks, max_episode_steps)
        self.goal_radius = goal_radius

        # if randomize_tasks:
        np.random.seed(1337)
        radius = 1.0
        angles = np.linspace(0, np.pi, num=n_tasks)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        goals = np.stack([xs, ys], axis=1)
        if randomize_tasks:
            np.random.shuffle(goals)
        goals = goals.tolist()

        self.goals = goals
        if isinstance(goal_idx, int):
            self.reset_task(goal_idx)
        else:
            self.reset_task(0)

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        #mask = (r >= -self.goal_radius).astype(np.float32)
        #mask = (r >= -self.goal_radius)
        #r = r * mask
        #return r
        
        if r >= self.goal_radius:
            # sparse_reward = (r - self.goal_radius) * (1/abs(self.goal_radius)) # normalize reward to [0, 1]
            sparse_reward = r + 1
        else:
            sparse_reward = r * 0
        return sparse_reward
        
    def reset_model(self):
        self._state = np.array([0, 0])
        self._step = 0
        return self._get_obs()

    def step(self, action):
        # print('goal', self._goal)
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        #if reward >= -self.goal_radius:
        #    sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d
        # return ob, sparse_reward, done, d
