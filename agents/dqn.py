"""
The code is taken and changed from Deep Reinforcement Learning Hands on book author Max Lapan

link: https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On
"""
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)


# Define Noisy Layer
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.01, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features,
                                                     in_features),
                                                    sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features,
                                                           in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,),
                                                      sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input_x):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input_x,
                        self.weight + self.sigma_weight * self.epsilon_weight.data,
                        bias)


# Define Q-Networks with Noisy Dense
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, n_hiddens=32):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            NoisyLinear(input_shape[0], n_hiddens),
            nn.ReLU(),
            NoisyLinear(n_hiddens, n_hiddens),
            nn.ReLU(),
            NoisyLinear(n_hiddens, n_actions)
        )

    def forward(self, x):
        x = x.float()
        return self.fc(x)

    def qvals(self, x):
        return self.forward(x)


# Define Dueling Networks
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions, n_hiddens=32):
        super(DuelingDQN, self).__init__()

        self.fc_adv = nn.Sequential(
            NoisyLinear(input_shape[0], n_hiddens),
            nn.ReLU(),
            NoisyLinear(n_hiddens, n_hiddens),
            nn.ReLU(),
            NoisyLinear(n_hiddens, n_actions)
        )
        self.fc_val = nn.Sequential(
            NoisyLinear(input_shape[0], n_hiddens),
            nn.ReLU(),
            NoisyLinear(n_hiddens, n_hiddens),
            nn.ReLU(),
            NoisyLinear(n_hiddens, 1)
        )

    def forward(self, x):
        fx = x.float()
        val = self.fc_val(fx)
        adv = self.fc_adv(fx)
        return val + adv - adv.mean()

    def qvals(self, x):
        return self.forward(x)


# Deep DQN with Dueling, Disributional DQN and Noisy Dense
# it includes the Convolution 1D layers, Max Pooling layers
class CNNDQN(nn.Module):
    def __init__(self, input_shape, n_actions, n_hiddens=32,
                 Vmin=-10, Vmax=10, n_atoms=51):
        super(CNNDQN, self).__init__()

        self.n_atoms = n_atoms
        DELTA_Z = (Vmax - Vmin) / (self.n_atoms - 1)

        self.conv = nn.Sequential(
            nn.Conv1d(input_shape[0], 32, 2),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            # nn.Conv1d(32, 32, 3),
            # nn.ReLU(),
            # nn.Conv1d(32, 32, 3),
            # nn.ReLU(),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_val = nn.Sequential(
            NoisyLinear(conv_out_size, n_hiddens),
            nn.ReLU(),
            NoisyLinear(n_hiddens, self.n_atoms)
        )

        self.fc_adv = nn.Sequential(
            NoisyLinear(conv_out_size, n_hiddens),
            nn.ReLU(),
            NoisyLinear(n_hiddens, n_actions * self.n_atoms)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax + DELTA_Z,
                                                      DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float()
        conv_out = self.conv(fx).view(batch_size, -1)
        val_out = self.fc_val(conv_out).view(batch_size, 1,
                                             self.n_atoms)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1,
                                             self.n_atoms)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + adv_out - adv_mean

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, self.n_atoms)).view(t.size())
