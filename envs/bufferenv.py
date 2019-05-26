import gym
import numpy as np
from gym.utils import seeding
from envs.serverpool import QueuingServerPoolWithLatency
from envs.states import State, State1D, Actions
import globalvars

np.random.seed(globalvars.GLOBAL_SEED)


# Define Environment
class BuffersEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        if 'bars_count' in kwargs:
            bars_count = kwargs['bars_count']
        else:
            bars_count = globalvars.DEFAULT_BAR_COUNTS
        if 'workload_type' in kwargs:
            workload_type = kwargs['workload_type']
        else:
            workload_type = globalvars.WORKLOAD_TYPE
        if 'FIS' in kwargs:
            FIS = kwargs['FIS']
        else:
            FIS = None

        if 'serverpool' in kwargs:
            self.serverpool = kwargs['serverpool']
        else:
            self.serverpool = QueuingServerPoolWithLatency(n=1, latency=10,
                                                           workload_type=workload_type)
        if 'state_1d' in kwargs:
            state_1d = kwargs['state_1d']
        else:
            state_1d = False

        if state_1d:
            self._state = State1D(self, bars_count=bars_count, FIS=FIS)
        else:
            self._state = State(self, bars_count=bars_count, FIS=FIS)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=self._state.shape,
                                                dtype=np.float32)
        self.seed(globalvars.GLOBAL_SEED)

    def reset(self):
        return self._state.reset()

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
