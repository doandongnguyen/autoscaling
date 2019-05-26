import os
import copy
import numpy as np
import random
import torch
import matplotlib
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from envs.bufferenv import BuffersEnv
from agents.memory import ExperienceBuffer
from agents.dqn import CNNDQN
from agents.agents import Agent
from agents.utils import calc_loss_probs
from BufferFIS import Buffer_Fis
import globalvars


np.random.seed(globalvars.GLOBAL_SEED)
random.seed(globalvars.GLOBAL_SEED)
torch.manual_seed(globalvars.GLOBAL_SEED)

device = "cuda"
PATH = ""
font = {'family': 'normal',
        'weight': 'normal',
        'size': 25}

matplotlib.rc('font', **font)
FIS = Buffer_Fis()
env = BuffersEnv(state_1d=True,
                 workload_type=globalvars.WORKLOAD_TYPE,
                 FIS=FIS
                 )
input_shape = env.observation_space.shape
action_size = env.action_space.n
print('Initializing DQN:')
MODEL_NAME = PATH + globalvars.DEFAULT_ENV_NAME + '-FDQN.dat'
net = CNNDQN(input_shape,
             action_size,
             n_hiddens=globalvars.N_HIDDENS,
             Vmin=globalvars.Vmin,
             Vmax=globalvars.Vmax,
             n_atoms=globalvars.N_ATOMS).to(device)
# Initialize target NN
tgt_net = copy.deepcopy(net)
print('Model: ', globalvars.MODEL_TYPE)
print('PATH: ', MODEL_NAME)
buffer = ExperienceBuffer(globalvars.REPLAY_SIZE)
agent = Agent(env, buffer)
optimizer = optim.Adam(net.parameters(),
                       lr=globalvars.LEARNING_RATE)


def training():
    total_rewards = []
    steps = 0
    best_mean_reward = None
    episode = 0
    start = datetime.now()
    print('start at: ', start)
    while True:
        steps += 1
        reward = agent.play_step(net=net, device=device)
        if reward is not None:
            episode += 1
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-50:])
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), MODEL_NAME)
                if best_mean_reward is not None:
                    epsilon = abs(best_mean_reward - mean_reward)
                best_mean_reward = mean_reward
            if best_mean_reward > 140.0:
                break
        if len(buffer) < globalvars.REPLAY_START_SIZE:
            continue
        if steps % globalvars.SYNC_TARGET == 0:
            tgt_net.load_state_dict(net.state_dict())
        optimizer.zero_grad()
        batch = buffer.sample(globalvars.BATCH_SIZE)
        loss_t = calc_loss_probs(batch, net, tgt_net,
                                 gamma=globalvars.GAMMA, device=device,
                                 Vmin=globalvars.Vmin,
                                 Vmax=globalvars.Vmax,
                                 N_ATOMS=globalvars.N_ATOMS)
        loss_t.backward()
        optimizer.step()
    print('Total time: ', datetime.now() - start)
    return


def evaluating():
    np.random.seed(globalvars.GLOBAL_SEED)
    random.seed(globalvars.GLOBAL_SEED)
    if os.path.exists(MODEL_NAME):
        net.load_state_dict(torch.load(MODEL_NAME))
        print('Load model successfully, ', MODEL_NAME)
    reward_list = []
    rewards = []
    queue = []
    nums = []

    for count in range(1):
        state = env.reset()
        total_reward = 0.0
        while True:
            state_v = torch.tensor(np.array([state], copy=False)).to(device)
            q_vals = net.qvals(state_v).data.cpu().numpy()
            action = np.argmax(q_vals, axis=1)[0]
            state, reward, done, _ = env.step(action)
            total_reward += reward
            queue.append(env.serverpool.queue/100)
            rewards.append(reward*10)
            nums.append(env.serverpool.n)
            if done:
                reward_list.append(total_reward)
                break
        print("Episode %d Total reward: %.2f" % (count+1, total_reward))

    plt.figure(figsize=(15, 10), dpi=120, facecolor='w', edgecolor='k')
    ax = plt.gca()
    ax2 = ax.twinx()
    plt.axis('normal')
    ax.plot(np.array(env._state.rates),
            label='Queue Percentages', color='blue')
    ax.set_ylabel('Percentages', fontsize=25, color='blue')
    ax.set_xlabel('Time', fontsize=25, color='black')
    ax.set_ylim(top=100)
    ax2.plot(env._state.nb_instances,
             label='No. of Instances', color='r')
    ax2.set_ylabel('No. of Instances', fontsize=25, color='r')
    ax2.set_ylim(top=10)
    plt.savefig('results.png')
    plt.show()


if __name__ == '__main__':
    evaluating()
