import numpy as np
import collections
import globalvars

np.random.seed(globalvars.GLOBAL_SEED)
Experience = collections.namedtuple('Experience',
                                    field_names=['state', 'action',
                                                 'reward', 'done',
                                                 'new_state'])


class ExperienceBuffer:
    """
        Define the Buffer to store experiences
    """
    def __init__(self, capacity):
        """
        :param capacity: maximum capacity of buffer
        """
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer),
                                   batch_size, replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)
