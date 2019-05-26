"""
The code is taken and changed from Deep Reinforcement Learning Hands on book author Max Lapan

link: https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On
"""

import enum
import numpy as np
import globalvars
from utils import utils
np.random.seed(globalvars.GLOBAL_SEED)


class Actions(enum.Enum):
    Skip = 0
    Increase = 1
    D_Increase = 2
    Decrease = 3
    D_Decrease = 4


class State:
    def __init__(self, server, **kwargs):
        """
        :param server:
        :param kwargs:
        """
        if 'bars_count' in kwargs:
            bars_count = kwargs['bars_count']
        else:
            bars_count = globalvars.DEFAULT_BAR_COUNTS
        if 'FIS' in kwargs:
            FIS = kwargs['FIS']
        else:
            FIS = None
        self.server = server
        self.bars_count = bars_count
        self.workloads = []
        self.buffers = []
        self.nb_instances = []
        self.rates = []
        self.rewards = []
        print('Initializing State with bar count=',
              self.bars_count)
        if FIS is not None:
            self.fis = FIS
            print('Initializing State with Fuzzy')
        else:
            self.fis = None

    def reset(self):
        """
        :return: The first state
        """
        self.workloads.clear()
        self.buffers.clear()
        self.nb_instances.clear()
        self.server.serverpool.reset()
        self.rewards.clear()
        self.rates.clear()
        nb_instance = self.server.serverpool.n
        means = []
        for i in range(10 + self.bars_count):
            rate = self.server.serverpool.work(nb_instance)
            result = self._monitoring(rate)
            means.append(rate)
            if result is None:
                break
        reward = np.mean(means[-1:])
        reward = self._buffer_reward(reward,
                                     nb_instance,
                                     Actions.Skip)
        self.rewards.append(reward)
        return self.encode()

    def _monitoring(self, rate):
        """
        :param rate: recorded the rate
        :return:
        """
        state = self.server.serverpool.monitoring()
        if state is None:
            return None
        w, b, n = state
        self.workloads.append(w)
        self.buffers.append(b)
        self.nb_instances.append(n)
        self.rates.append(rate)
        return 0

    @property
    def shape(self):
        """
        :return: return the shape of state
        """
        if self.fis is not None:
            return (self.bars_count*self.fis.rules.get_number_of_rules(), )
        else:
            return (2 * self.bars_count, )

    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=(2*self.bars_count,),
                         dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count, 0):
            res[shift] = self.workloads[bar_idx]
            shift += 1
            res[shift] = self.rates[bar_idx]
            shift += 1
        if self.fis is not None:
            vals = self.to_fuzzy(res)
            vals = vals.flatten()
            return vals
        else:
            return res

    def to_fuzzy(self, res):
        """
        :param res: the input variables
        :return: fuzzified input variables
        """
        vals = []
        for i in range(self.bars_count):
            state = res[2*i:2*(i+1)]
            truth_vals = self.fis.get_truth_values(state)
            vals.append(truth_vals)
        return np.array(vals)

    def step(self, action):
        """
        :param action: interact to environment with action
        :return: the reward and done
        """
        assert isinstance(action, Actions)
        done = False
        nb_instance = self.server.serverpool.n
        if action == Actions.Skip:
            nb_instance = self.server.serverpool.n
        elif action == Actions.Increase:
            nb_instance += 1
        elif action == Actions.Decrease:
            nb_instance -= 1
        elif action == Actions.D_Increase:
            nb_instance += 2
        elif action == Actions.D_Decrease:
            nb_instance -= 2
        if nb_instance > self.server.serverpool.maximum_instances:
            nb_instance = self.server.serverpool.maximum_instances
        if nb_instance <= 0:
            nb_instance = 1
        means = []
        for i in range(10 + self.bars_count):
            rate = self.server.serverpool.work(nb_instance)
            result = self._monitoring(rate)
            if result is None:
                done = True
                break
            means.append(rate)
        if len(means) > 0:
            reward = np.mean(means[-1:])
            reward = self._buffer_reward(ht=reward,
                                         nb_instance=nb_instance,
                                         action=action)
        else:
            reward = 0.
        self.rewards.append(reward)
        return reward, done

    def _buffer_reward(self, ht, nb_instance, action):
        """
        Calculating reward
        the purpose is to keep buffer in the safe zone
        from (lo_threshold to hi_threshold)
        reward value is in [-1, 1]
        -1 -----> 0 ----> 1 ---> 0 ---> -1
        max-----> hi----> mid--->low--->min = 0
        results = w1*BU(t) + w2*(VM(t)/VM_max)
        :param ht: the current buffer value
        :param nb_instance:  number of current instance
        :param action: action have been taken
        :return: value of reward
        """
        max_threshold = 100.
        hi_threshold = 70.
        mid_threshold = 50.
        lo_threshold = 30.
        safe_zone = (hi_threshold - lo_threshold)/2
        w1, w2 = 0.8, 0.2
        if ht > hi_threshold:
            reward = -(ht-hi_threshold)/(max_threshold-hi_threshold)
        elif ht < lo_threshold:
            reward = -(lo_threshold-ht)/lo_threshold
            reward = w1*reward + w2*(1-nb_instance/self.server.serverpool.maximum_instances)
        else:
            if ht >= mid_threshold:
                reward = (hi_threshold-ht)/safe_zone
            else:
                reward = (ht-lo_threshold)/safe_zone
            reward = w1*reward + w2*(1-nb_instance/self.server.serverpool.maximum_instances)
        return reward


class State1D(State):
    """
    State with shape suitable for 1D convolution
    """
    @property
    def shape(self):
        if self.fis is not None:
            return (self.fis.rules.get_number_of_rules(),
                    self.bars_count)
        else:
            return 2, self.bars_count

    def encode(self):
        """
            To encode the data to 1D array or fuzzy data
        :return: the state of environment
        """
        res = np.zeros(shape=(2, self.bars_count),
                       dtype=np.float32)
        ofs = self.bars_count
        N = 5
        if self.fis is not None:
            x = self.workloads[-ofs - N + 1:]
            res[0] = utils.running_mean(x, N)
            x = self.rates[-ofs - N + 1:]
            res[1] = utils.running_mean(x, N)
            # res[2] = self.nb_instances[-ofs:]
            vals = self.to_fuzzy(res)
            return vals.transpose()
        else:
            res[0] = self.workloads[-ofs:]
            res[1] = self.rates[-ofs:]
            # res[2] = self.nb_instances[-ofs:]
            return res

    def to_fuzzy(self, res):
        rows, cols = res.shape
        vals = []
        for i in range(cols):
            state = (res[0, i], res[1, i])  # , res[2, i])
            truth_vals = self.fis.get_truth_values(state)
            vals.append(truth_vals)
        return np.array(vals)

