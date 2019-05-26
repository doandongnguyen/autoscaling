import torch
import numpy as np
from agents.memory import Experience
import globalvars

np.random.seed(globalvars.GLOBAL_SEED)


class Agent:
    def __init__(self, env, exp_buffer):
        """

        :param env:  Environment variables
        :param exp_buffer: Experience buffer
        """
        self.env = env
        self.exp_buffer = exp_buffer
        self.state = None
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, device="cpu"):
        done_reward = None
        # Choose actions
        state_a = np.array([self.state], copy=False)
        state_v = torch.tensor(state_a).to(device)
        q_vals_v = net.qvals(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.item())
        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        if self.total_reward >= 1000:
            is_done = True

        exp = Experience(self.state, action,
                         reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward
